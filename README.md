# arabidopsis-population-aware-benchmark

Code and processed data for a population-aware benchmark of genomic and
transcriptomic prediction in *Arabidopsis thaliana*.

The benchmark compares genomic (G), transcriptomic (E), and combined (G+E)
prediction with kernel ridge regression (KRR), Lasso, and histogram gradient
boosting (HGB) under random and leave-population-out (LOPO) cross-validation,
across 27 phenotypes from eight AraPheno studies. The matched panel comprises
654 accessions with 1,135 genomes and 727 leaf transcriptomes available in the
source cohorts.

## Repository layout

```
scripts/          analysis pipeline (numbered in run order) + benchmark_lib.py
data_processed/   QC'd genotype/expression matrices, kernels, PCs, phenotypes,
                  population labels, and exported CV splits
results/          benchmark outputs: per-trait accuracy, heritability, transfer,
                  permutation controls, enrichment, sensitivity analyses
run_pipeline.sh   one-command full rebuild
requirements.txt  pinned Python dependencies
```

## Data sources

Raw inputs are not redistributed here. Download them from the original
providers:

| Input | Source |
|---|---|
| 1001 Genomes imputed SNP matrix and IBS kinship | https://1001genomes.org |
| AraPheno study and phenotype tables | https://arapheno.cebitec.uni-bielefeld.de |
| 1001 Transcriptomes normalized counts (GSE80744) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80744 |
| TAIR10 GFF3, GO annotation (GAF), GO ontology (OBO) | https://www.arabidopsis.org |

Place the raw files under `data/` in the repository root (or point the
`ARABIDOPSIS_DATA_DIR` environment variable at an existing copy), using this
layout:

```
data/
  01_genotype_1001G/1001_SNP_MATRIX/imputed_snps_binary.hdf5
  01_genotype_1001G/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5
  02_phenotype_AraPheno/study_list.json
  02_phenotype_AraPheno/phenotype_list.json
  03_transcriptome_1001T/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz
  05_annotation_GO_KEGG/tair.gaf.gz
  05_annotation_GO_KEGG/go-basic.obo
  06_genome_TAIR10/Arabidopsis_thaliana.TAIR10.63.gff3.gz
```

## Setup

Python 3.13 with the pinned packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The pipeline runs on a desktop CPU; no GPU is required. The prediction steps
(09, 17, 18) are the most compute-intensive and may take several hours.

## Running

Two entry points:

1. **From the included processed data** (no raw downloads needed): run
   `scripts/09_prediction.py` onward. All downstream scripts read
   `data_processed/` and write to `results/`.

2. **Full rebuild from raw data**: run the whole pipeline in order:

```
bash run_pipeline.sh
```

or set `PYTHON` to a specific interpreter, e.g.
`PYTHON=/path/to/python bash run_pipeline.sh`. Per-step logs are written to
`logs/`. Figures are written to `figures/` and supplementary tables to
`manuscript/tables/` (both are git-ignored build outputs).

Run order: 00 environment provenance, 01 QC, 02 genotype build, 03 phenotypes,
04 heritability, 05 trait selection, 06 expression, 07 populations,
08 trait finalization, 09 prediction (matched panel, then full genotype panel),
10 cross-study transfer, 11 permutation controls, 12 GO enrichment,
15 split export, 17 kernel-weight recovery, 18 sensitivity analyses,
13 statistical analysis, 14 figures, 16 tables, 19 sensitivity tables.

## Key parameters

MAF >= 0.05, 5-kb thinning, LD pruning (200-SNP window, |r| <= 0.5); 50 genotype
PCs; six population clusters (rare groups merged at n < 10); traits with
n >= 250 and REML h2 in [0.2, 0.95]; expression log2(count+1) with gene-wise
z-scores; KRR alpha grid 0.01-100 and G/E kernel weight grid 0-1 tuned by inner
3-fold CV; primary metric is the Pearson r of out-of-fold predictions.

## Citation

If you use this benchmark, please cite the repository and the accompanying
manuscript:

> Tu L, Peng Q. A population-aware benchmark of genomic and transcriptomic
> prediction across 27 *Arabidopsis thaliana* phenotypes. Manuscript in
> preparation.

## Contact

Qi Peng (corresponding author) — petersky0316@hbeu.edu.cn
School of Physics and Electronic Information Engineering, Hubei Engineering
University, Xiaogan, Hubei, China