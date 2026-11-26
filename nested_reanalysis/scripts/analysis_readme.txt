Arabidopsis genomic prediction: Plant Gene revision

Article: Genetic differentiation and phenotype reference coverage in Arabidopsis genomic prediction

This archive supplies the fully nested retrospective reanalysis and the additional outer-fold-size-matched random sensitivity. It does not contain raw sequencing reads, new phenotypes, editorial working notes or manuscript build scripts. Earlier analysis filenames are retained for provenance, not as a claim that this revision has been submitted to another journal.

Environment used
Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2, Matplotlib 3.10.6. Numerical linear algebra uses one thread per process; prediction commands use two workers. No GPU is required.

Files
scripts/biosystems_lib.py: trait matching, group merging, partitions, training-only fitted transformations, KRR, metrics and Hudson differentiation.
scripts/24_nested_validation.py: primary 27-trait random/group-held-out predictions, group-mean and group-centered diagnostics.
scripts/25_nested_sampling.py: G-only retrospective zero/candidate-half label inclusion with added and constant-size replacement designs.
scripts/26_statistical_synthesis.py: paired study-block synthesis, rank-residual diagnostics and source omissions.
scripts/28_size_matched_validation.py: five-repeat G random partitions exactly matching original LOPO outer train/test counts; separate all-27 and original-always-grouped-inner-20 summaries.
scripts/27_revision_figures.py: four main and four supplementary data-derived plots. This requires the supplied numerical results and exact inputs. Create empty manuscript/figures/ and work/ directories before running it in an extracted code-only copy. It writes vector PDFs and preview PNGs, plus 600-dpi uncompressed TIFFs for main artwork.
scripts/test_nested_validation.py: KRR agreement and fitted-transform isolation tests on synthetic data.
scripts/verify_science.py: independent verification of the supplied fold-size sensitivity from saved prediction bundles, reconstructed CRC32 assignments, fold counts, training-ID hashes, metrics, five-repeat means, six paired/block summaries and input hashes.
results/nested/: 81 primary accession-level NPZ prediction bundles, raw/aggregated metrics and fold details.
results/sampling/: complete numerical records and 67 target-unit JSON files with exact evaluation/candidate/removal accession IDs for ten splits each.
results/synthesis/: effect, source, omission, differentiation, decomposition and tuning tables; analysis_record.json records primary methods and input/code hashes.
results/size_matched/: 135 new prediction bundles, raw folds/metrics, five-repeat per-trait averages, paired/source/omission tables, tuning counts, initial-group trait coverage and a separate analysis_record.json.
results/legacy/: earlier global/outer-only summaries, clearly separated historical sensitivities rather than fully nested primary outputs.
verification/: numerical and artwork audit records supplied for this revision. Project-specific checks of preserved originals are included as reports, not portable reconstruction prerequisites.
archive_sha256.json: SHA256 for every other member of this archive.

Inputs
The code and numerical results in this archive are also deposited under nested_reanalysis/ in https://github.com/TTFriday/arabidopsis-population-aware-benchmark, together with inputs/sha256.json. Seven of the eight hashed processed inputs are byte-identical to files already published in that repository under data_processed/, so they are not duplicated here; the remaining input, inputs/expr_log2_raw.npy, is the untransformed log2 expression matrix retained by the corresponding author. Extract inputs/ alongside scripts/ and results/. inputs/sha256.json and the two analysis records identify the exact files. Public upstream resources and limitations are described in the article.

Default layout
scripts/
results/
inputs/data_processed/geno_matrix.npy
inputs/data_processed/accessions.csv
inputs/data_processed/expr_accessions.csv
inputs/data_processed/phenotypes_long.csv
inputs/data_processed/pop6.csv
inputs/data_processed/PCs.csv
inputs/data_processed/benchmark_traits.csv
inputs/expr_log2_raw.npy

Run from the directory containing these folders
python scripts/test_nested_validation.py
python scripts/24_nested_validation.py --workers 2
python scripts/25_nested_sampling.py --workers 2
python scripts/26_statistical_synthesis.py
python scripts/28_size_matched_validation.py --workers 2
python scripts/verify_science.py
python scripts/27_revision_figures.py

Commands replace matching numerical output files. Use a separate extracted copy to preserve supplied results. Synthetic tests do not need biological inputs. The sensitivity requires original LOPO predictions and fold records; synthesis uses supplied numerical results but also verifies exact input hashes. Its intervals condition on five-repeat averages and do not resplit in each block resample.

To use inputs elsewhere, set ARABIDOPSIS_PROCESSED to the processed panel-file directory and ARABIDOPSIS_EXPRESSION_LOG to the expression NPY file. Upstream normalization, marker ascertainment, archived PCs/groups and trait selection remain fixed.

Interpretation
Raw accession replicate means are evaluated. Marker/expression transformations, outcome clipping/scaling and hyperparameter selection are nested within each inner-training fold and outer-training set. LOPO requests grouped inner tuning where possible and reports the one-group fallback. All 425 new matched-random folds actually use grouped inner tuning. Matching counts and a tuning rule does not match genetic composition or exact inner partitions and does not isolate a causal structure effect.

The separate label experiment uses random inner tuning and reveals existing records, not new phenotyping. Trait/group units and studies share accessions. Repeated partitions are not independent biological replicates. Source-block intervals are empirical stability ranges, not population confidence guarantees. Source omissions change summaries, not prediction training. Group-centered scores use observed evaluation outcomes diagnostically. Analyses were developed after examining earlier results and were not preregistered.
