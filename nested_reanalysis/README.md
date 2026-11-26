# nested_reanalysis

Fully nested reanalysis of the 27-trait *Arabidopsis thaliana* benchmark, plus the
outer-fold-size-matched random sensitivity, the retrospective label-inclusion
experiment and the independent numerical verifier. This directory reproduces the
supplementary code and result tables supplied as SM3 with the manuscript; the
archived pipeline that produced the earlier tables in this repository is left
untouched.

## Why a second analysis tree exists

The archived scripts in `../scripts` select markers, scale features, clip and
scale outcomes and tune hyperparameters using information from the whole
cohort before the outer split is applied. The scripts here refit every one of
those steps inside the inner-training fold and again on the outer-training set,
and evaluate against raw replicate-mean phenotypes. Because the preprocessing
is nested, the accuracy estimates in this directory differ from the archived
ones and are the ones reported in the article.

The archived summary tables are not duplicated here. They already live in
`../results/analysis`, `../results/augmentation`, `../results/diversity` and
`../results/strict_pipeline`, and SM3 ships them under `results/legacy/` with
identical row content (SM3 copies carry CRLF line endings).

## Layout

```
scripts/biosystems_lib.py             shared library (trait matching, partitions,
                                      training-only fitted transforms, KRR, metrics,
                                      Hudson differentiation)
scripts/24_nested_validation.py       primary random and leave-population-out predictions
scripts/25_nested_sampling.py         retrospective label inclusion (added / size-matched)
scripts/26_statistical_synthesis.py   paired study-block synthesis, rank residuals
scripts/28_size_matched_validation.py 5-repeat random folds matching LOPO fold sizes
scripts/test_nested_validation.py     synthetic KRR and transform-isolation tests
scripts/verify_science.py             independent audit of the shipped fold-size results
scripts/analysis_readme.txt           the archive-level notes shipped with SM3
results/nested/                       81 accession-level prediction bundles and metrics
results/sampling/                     label-inclusion tables and 67 unit split records
results/size_matched/                 135 prediction bundles, 425 fold assignments,
                                      per-trait averages, paired/block summaries
results/synthesis/                    effect, source, omission and tuning tables
inputs/sha256.json                    hashes of the eight processed inputs used
sha256.json                           hashes of every file in this directory
```

Figure rendering and the revision-time delivery audit reports are supplied in
SM3 only, consistent with this repository's policy of not publishing manuscript
figures.

## Environment

Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2,
Matplotlib 3.10.6. Single-threaded linear algebra per process; prediction
scripts accept `--workers 2`. No GPU is required.

## Reconstructing the inputs

`inputs/sha256.json` lists the eight processed inputs. Seven are byte-identical
to files already published in this repository, so they are not duplicated here:

| Required path | Published file |
|---|---|
| `inputs/data_processed/geno_matrix.npy` | `data_processed/geno_matrix.npy` |
| `inputs/data_processed/accessions.csv` | `data_processed/accessions.csv` |
| `inputs/data_processed/expr_accessions.csv` | `data_processed/expr_accessions.csv` |
| `inputs/data_processed/phenotypes_long.csv` | `data_processed/phenotypes_long.csv` |
| `inputs/data_processed/pop6.csv` | `data_processed/pop6.csv` |
| `inputs/data_processed/PCs.csv` | `data_processed/PCs.csv` |
| `inputs/data_processed/benchmark_traits.csv` | `data_processed/benchmark_traits.csv` |
| `inputs/expr_log2_raw.npy` | retained by the corresponding author |

From a clone of this repository, run inside `nested_reanalysis/`:

```
mkdir -p inputs && cp -r ../data_processed inputs/data_processed
# place expr_log2_raw.npy in inputs/, then check the eight hashes:
python - <<'PY'
import hashlib, json, pathlib
for name, digest in json.loads(pathlib.Path('inputs/sha256.json').read_text()).items():
    print(name, hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == digest)
PY
```

## Running

From inside `nested_reanalysis/`:

```
python scripts/test_nested_validation.py
python scripts/verify_science.py
python scripts/24_nested_validation.py --workers 2
python scripts/25_nested_sampling.py --workers 2
python scripts/26_statistical_synthesis.py
python scripts/28_size_matched_validation.py --workers 2
```

`test_nested_validation.py` and `verify_science.py` need no biological input
files beyond the shipped results plus `inputs/data_processed/benchmark_traits.csv`.
The regeneration scripts overwrite the shipped tables, so run them in a separate
copy if you want to keep the published results intact.

`verify_science.py` works only from the shipped prediction bundles. It
reconstructs the CRC32-derived fold assignments for all 425 folds, checks
accession/phenotype/cluster identity against the original LOPO bundles,
recomputes every fold and pooled metric at absolute tolerance 1e-12 with no
relative slack, recomputes the five-repeat means and the six paired and
study-block bootstrap summaries, and asserts that every new random fold used
grouped inner tuning. Only then does it hash the eight inputs and write
`results/size_matched/independent_verification.json`, which is also printed.
Without `expr_log2_raw.npy` the numeric audit still runs to completion but the
run ends at that last hash step and no report file is produced; the checked
copy of this directory reproduced the shipped report byte for byte once all
eight inputs were present.

## Interpretation limits

Matching outer-fold counts and a tuning rule does not match genetic composition
or exact inner partitions, so the fold-size sensitivity does not isolate a
causal effect of population structure. The label experiment reuses existing
records rather than new phenotyping. Trait/group units and studies share
accessions, and repeated partitions are not independent biological replicates,
so source-block intervals are empirical stability ranges rather than population
confidence guarantees. These analyses were developed after examining earlier
results and were not preregistered.

The module name `biosystems_lib.py` and the numbered script prefixes are retained
from earlier working revisions for provenance and for the code hashes recorded
in `results/*/analysis_record.json`; renaming them would break those records.
