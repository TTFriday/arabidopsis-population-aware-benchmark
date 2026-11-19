"""Paths and helpers shared by the follow-up analyses (inputs are read-only)."""
import os, sys
sys.dont_write_bytecode = True
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.environ.get("ARABIDOPSIS_BENCHMARK_ROOT", ROOT)
NEW = os.environ.get("GRCE_FOLLOWUP_ROOT", ROOT)
DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(ROOT, "data"))
RES = os.path.join(NEW, "results")
WORK = os.path.join(NEW, "work")
sys.path.insert(0, os.path.join(ORIG, "scripts"))

import numpy as np

ALPHAS = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0])
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def hudson_fst(p1, p2, n1, n2):
    """Ratio-of-averages Hudson Fst (Bhatia et al. 2013); n = haplotypes (inbred accessions)."""
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    return float(num.sum() / den.sum())


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, p[idx] * (len(p) - rank))
        adj[idx] = min(run, 1.0)
    return adj