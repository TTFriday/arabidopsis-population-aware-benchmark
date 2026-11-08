"""Export sensitivity analyses (script 18) as manuscript supplementary tables S6-S9."""
import os, sys, json
from importlib import import_module
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import OUT
summarize_clusters = import_module("18_sensitivity").summarize_clusters

RES = os.path.join(OUT, "results")
SENS = os.path.join(RES, "sensitivity")
TAB = os.path.join(OUT, "manuscript", "tables")
MODEL = {"kridge": "KRR", "lasso": "Lasso", "hgb": "HGB"}
FEAT = {"G": "G", "E": "E", "GE": "G+E"}
CV = {"random": "Random", "lopo": "LOPO"}


def write(df, name):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    print("wrote", name, df.shape)


# S6 metric robustness
m = pd.read_csv(os.path.join(SENS, "metric_summary.csv"))
m["Model"] = m.Model.map(MODEL); m["Features"] = m.Features.map(FEAT); m["CV"] = m.CV.map(CV)
m = m[["Model", "Features", "CV", "Median Pearson r", "Median Spearman rho", "Median R2", "Median MAE"]]
m = m.sort_values(["Model", "Features", "CV"]).reset_index(drop=True)
write(m, "tableS6_metric_robustness")

# S7 domain accuracy
d = pd.read_csv(os.path.join(SENS, "domain_summary.csv"))
d = d.rename(columns={"n_traits": "n traits"})
d = d[["Domain", "n traits", "KRR G random", "KRR G+E random", "KRR G LOPO", "KRR G+E LOPO",
       "Median gap G", "Median gap G+E"]]
write(d, "tableS7_domain_accuracy")

# S8 cluster sensitivity: exclude invalid outer LOPO and match random-CV traits.
c = summarize_clusters(pd.read_csv(os.path.join(SENS, "cluster_per_trait.csv"), float_precision="round_trip"),
                       pd.read_csv(os.path.join(RES, "analysis", "per_trait.csv"), float_precision="round_trip"))
c.to_csv(os.path.join(SENS, "cluster_summary.csv"), index=False)
c["Features"] = c.Features.map(FEAT)
write(c, "tableS8_cluster_sensitivity")

# S9 expression residualization
orig = pd.read_csv(os.path.join(RES, "analysis", "per_trait.csv"))
o = {"E": {"random": orig.kridge__E__random.median(), "lopo": orig.kridge__E__lopo.median()},
     "GE": {"random": orig.kridge__GE__random.median(), "lopo": orig.kridge__GE__lopo.median()}}
r = pd.read_csv(os.path.join(SENS, "expr_residual_summary.csv"))
rows = []
for row in r.itertuples():
    ov = o[row.Features][row.CV]
    rows.append({"Features": FEAT[row.Features], "CV": CV[row.CV],
                 "Original median r": ov, "Residualized median r": row.r, "Delta": row.r - ov})
s9 = pd.DataFrame(rows)
write(s9, "tableS9_expr_residualization")
info = json.load(open(os.path.join(SENS, "residualization_info.json")))
print("variance removed overall %.3f median gene %.3f n_shared %d" %
      (info["variance_removed_overall"], info["variance_removed_median_gene"], info["n_shared"]))
print("\nS6 Spearman GE-vs-G tests:")
print(pd.read_csv(os.path.join(SENS, "metric_spearman_tests.csv")).to_string(index=False))