# arabidopsis-population-aware-benchmark

Code and processed data for a population-aware benchmark of genomic and
transcriptomic prediction in *Arabidopsis thaliana*, together with follow-up
analyses of genetic differentiation and target-group phenotyping for
germplasm evaluation.

The benchmark compares genomic (G), transcriptomic (E), and combined (G+E)
prediction with kernel ridge regression (KRR), Lasso, and histogram gradient
boosting (HGB) under random and leave-population-out (LOPO) cross-validation,
across 27 phenotypes from eight AraPheno studies. The matched panel comprises
654 accessions with 1,135 genomes and 727 leaf transcriptomes available in the
source cohorts.

## Repository layout

```
scripts/          analysis pipeline (numbered in run order) + benchmark_lib.py
                  + grce_common.py and follow-up analyses 20-22
data_processed/   QC'd genotype/expression matrices, kernels, PCs, phenotypes,
                  population labels, and exported CV splits
results/          benchmark outputs: per-trait accuracy, heritability, transfer,
                  permutation controls, enrichment, sensitivity analyses,
                  out-of-fold predictions (full/oof/), and follow-up tables
                  (diversity/, augmentation/, strict_pipeline/);
                  archived_tables/ holds the numerical manuscript tables
run_pipeline.sh   one-command full rebuild
nested_reanalysis/ fully nested reanalysis of the 27 traits plus the
                  fold-size-matched sensitivity; supplies the manuscript's
                  primary estimates (own README, tables and verifier)
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
(09, 17, 18) and the augmentation step (21) are the most compute-intensive and
may take several hours.

## Running

Two entry points:

1. **From the included processed data** (no raw downloads needed): run
   `scripts/09_prediction.py` onward. All downstream scripts read
   `data_processed/` and write to `results/`. Step 22 additionally needs the
   raw GSE80744 count table (see `ARABIDOPSIS_DATA_DIR`); it is skipped
   automatically when that file is absent.

2. **Full rebuild from raw data**: run the whole pipeline in order:

```
bash run_pipeline.sh
```

or set `PYTHON` to a specific interpreter, e.g.
`PYTHON=/path/to/python bash run_pipeline.sh`. Per-step logs are written to
`logs/`. Figure and manuscript-table generation scripts are not part of this
repository; the archived numerical tables are provided under
`results/archived_tables/`.

Run order: 00 environment provenance, 01 QC, 02 genotype build, 03 phenotypes,
04 heritability, 05 trait selection, 06 expression, 07 populations,
08 trait finalization, 09 prediction (matched panel, then full genotype panel),
10 cross-study transfer, 11 permutation controls, 12 GO enrichment,
15 split export, 17 kernel-weight recovery, 18 sensitivity analyses,
13 statistical analysis, 20 diversity/relatedness diagnostics,
21 target-group augmentation, 22 fold-internal preprocessing sensitivity.

## Follow-up analyses (genetic-resources revision)

- `20_diversity_relatedness.py`: group gene diversity, polymorphism and Hudson
  Fst on all 1,135 genotyped accessions; test-to-training Fst and maximum
  genomic relationship per trait x held-out group, joined to within-group
  accuracy; accession-level accuracy by relationship tertile. Reads the
  out-of-fold predictions in `results/full/oof/`.
- `21_augmentation.py [nproc]`: target-group phenotyping experiment. Each
  eligible trait x group (>= 20 accessions) is split into a fixed evaluation
  half and a candidate pool; nested draws add 0/10/25/50% of the group, with a
  constant-training-size design where feasible. Training-only phenotype
  clipping; scores against raw evaluation means. Default 6 processes.
- `22_strict_pipeline.py [nproc]`: KRR sensitivity with marker/expression/
  phenotype preprocessing fitted inside each outer training fold only,
  starting from the raw log2(normalized count + 1) matrix. Default 5 processes.

Paths resolve relative to the repository root. Overrides:
`ARABIDOPSIS_BENCHMARK_ROOT` (benchmark checkout providing `scripts/`,
`data_processed/`, `results/`), `GRCE_FOLLOWUP_ROOT` (output root for
`results/` and `work/`), `ARABIDOPSIS_DATA_DIR` (raw input directory).
Intermediate caches are written to `work/` (git-ignored).

## Key parameters

MAF >= 0.05, 5-kb thinning, LD pruning (200-SNP window, |r| <= 0.5); 50 genotype
PCs; six population clusters (rare groups merged at n < 10); traits with
n >= 250 and REML h2 in [0.2, 0.95]; expression log2(count+1) with gene-wise
z-scores; KRR alpha grid 0.01-100 and G/E kernel weight grid 0-1 tuned by inner
3-fold CV; primary metric is the Pearson r of out-of-fold predictions.

## Nested reanalysis

`nested_reanalysis/` holds the code and result tables supplied as supplementary
material SM3. Unlike `22_strict_pipeline.py`, which fits preprocessing inside
each outer training fold, these scripts refit marker filtering, feature
scaling, outcome clipping/scaling and hyperparameter selection inside every
inner tuning fold and refit on the outer training set, then score raw
replicate-mean phenotypes. They cover all 27 traits under both random
five-fold and leave-population-out validation, add the outer-fold-size-matched
random sensitivity and the retrospective label-inclusion experiment, and
include `scripts/verify_science.py`, which re-derives all 425 fold assignments
and every reported metric from the shipped prediction bundles without refitting
the models. See its README for input assembly, run order and the limits of
interpretation.

## Citation

If you use this benchmark, please cite the repository and the accompanying
manuscript:

> Tu L, Peng Q. Genetic differentiation and targeted phenotyping in genomic
> prediction of *Arabidopsis thaliana* germplasm. Manuscript in preparation.

## Contact

Qi Peng (corresponding author) — petersky0316@hbeu.edu.cn
School of Physics and Electronic Information Engineering, Hubei Engineering
University, Xiaogan, Hubei, China