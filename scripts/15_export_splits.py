"""Export all CV splits and panel membership for reproducibility."""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import Panel, make_splits, OUT, PROC

P = Panel()
traits = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
sdir = os.path.join(PROC, "splits"); os.makedirs(sdir, exist_ok=True)
rows = []
for r in traits.itertuples():
    ids, y, cl = P.trait(r.study_id, r.trait)
    cl, mapping = P.merge_rare(ids, cl)
    pd.DataFrame({"accession_id": ids, "cluster": cl}).to_csv(
        os.path.join(sdir, f"{r.trait_id}__panel.csv"), index=False)
    for cv in ["random", "lopo"]:
        reps = [0, 1] if cv == "random" else [0]
        for rep in reps:
            splits = make_splits(cl, cv, seed=rep + 1)
            out = []
            for k, (tr, te) in enumerate(splits):
                for i in te:
                    out.append((ids[i], k, "test"))
                for i in tr:
                    out.append((ids[i], k, "train"))
            pd.DataFrame(out, columns=["accession_id", "fold", "role"]).to_csv(
                os.path.join(sdir, f"{r.trait_id}__{cv}__r{rep}.csv"), index=False)
    rows.append({"trait_id": r.trait_id, "n": len(ids), "clusters": ";".join(map(str, mapping.values()))})
pd.DataFrame(rows).to_csv(os.path.join(sdir, "panel_summary.csv"), index=False)
print("exported", len(traits), "traits to", sdir)
