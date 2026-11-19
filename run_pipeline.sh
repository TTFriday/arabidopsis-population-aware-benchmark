#!/usr/bin/env bash
# Full benchmark pipeline. Usage: bash run_pipeline.sh
# Optional: PYTHON=/path/to/python bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs results work

run() {
    echo "[$(date '+%H:%M:%S')] $*"
    "$@"
}

run "$PY" scripts/00_environment.py        > logs/00_environment.log 2>&1
run "$PY" scripts/01_qc.py                 > logs/01_qc.log 2>&1
run "$PY" scripts/02_build_genotype.py     > logs/02_build_genotype.log 2>&1
run "$PY" scripts/03_phenotypes.py         > logs/03_phenotypes.log 2>&1
run "$PY" scripts/04_heritability.py       > logs/04_heritability.log 2>&1
run "$PY" scripts/05_select_traits.py      > logs/05_select_traits.log 2>&1
run "$PY" scripts/06_expression.py         > logs/06_expression.log 2>&1
run "$PY" scripts/07_populations.py        > logs/07_populations.log 2>&1
run "$PY" scripts/08_finalize_traits.py    > logs/08_finalize_traits.log 2>&1
run "$PY" scripts/09_prediction.py --repeats 2 --tag full > logs/09_full.log 2>&1
run "$PY" scripts/09_prediction.py --fullpanel 1 --models kridge --features G --cvs random,lopo --repeats 2 --tag fullpanel > logs/09_fullpanel.log 2>&1
run "$PY" scripts/10_transfer.py           > logs/10_transfer.log 2>&1
run "$PY" scripts/11_controls.py           > logs/11_controls.log 2>&1
run "$PY" scripts/12_enrichment.py         > logs/12_enrichment.log 2>&1
run "$PY" scripts/15_export_splits.py      > logs/15_export_splits.log 2>&1
run "$PY" scripts/17_weights.py            > logs/17_weights.log 2>&1
run "$PY" scripts/18_sensitivity.py        > logs/18_sensitivity.log 2>&1
run "$PY" scripts/13_analysis.py           > logs/13_analysis.log 2>&1
run "$PY" scripts/20_diversity_relatedness.py > logs/20_diversity.log 2>&1
run "$PY" scripts/21_augmentation.py 6     > logs/21_augmentation.log 2>&1
RAW="${ARABIDOPSIS_DATA_DIR:-data}/03_transcriptome_1001T/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz"
if [ -f "$RAW" ]; then
    run "$PY" scripts/22_strict_pipeline.py 5 > logs/22_strict.log 2>&1
else
    echo "Skipping 22_strict_pipeline.py: raw GSE80744 counts not found (set ARABIDOPSIS_DATA_DIR)."
fi
echo "Pipeline finished."