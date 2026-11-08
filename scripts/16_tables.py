"""Assemble manuscript tables (CSV + Markdown) from analysis outputs."""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import OUT, PROC

RES = os.path.join(OUT, "results")
ANA = os.path.join(RES, "analysis")
TAB = os.path.join(OUT, "manuscript", "tables")
os.makedirs(TAB, exist_ok=True)


def write(df, name, cols=None, fmt=None, caption=None):
    if cols is not None:
        df = df[cols]
    path = os.path.join(TAB, name)
    df.to_csv(path + ".csv", index=False)
    md = df.copy()
    if fmt:
        for c, f in fmt.items():
            if c in md:
                md[c] = md[c].map(lambda v: f.format(v) if pd.notna(v) and isinstance(v, (int, float, np.floating)) else v)
    lines = ["| " + " | ".join(md.columns) + " |", "|" + "|".join(["---"] * len(md.columns)) + "|"]
    for _, r in md.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    header = f"# {caption}\n\n" if caption else ""
    with open(path + ".md", "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")
    print("wrote", path + ".csv")


def main():
    meta = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    per = pd.read_csv(os.path.join(ANA, "per_trait.csv"))
    summ = pd.read_csv(os.path.join(ANA, "summary_by_combo.csv"))
    tests = pd.read_csv(os.path.join(ANA, "tests.csv"))

    t1 = meta[["trait_id", "study_id", "study_name", "trait", "group", "n", "n_expr", "h2", "mean", "sd"]].copy()
    t1.columns = ["Trait ID", "Study", "Study name", "AraPheno trait", "Domain", "n genotyped",
                  "n matched", "h2 (REML)", "Mean", "SD"]
    write(t1, "table1_trait_panel", caption="Table 1. The 27-trait benchmark panel.")

    t2 = summ.copy()
    t2.columns = ["Model", "Features", "CV", "n traits", "Mean r", "Median r", "SD", "Q1", "Q3",
                  "Min", "Max", "Fraction r>0.1", "Fraction r>0.3", "Fraction r>0.5"]
    write(t2, "table2_accuracy_summary", caption="Table 2. Predictive accuracy across the 27-trait panel.",
          fmt={c: "{:.3f}" for c in ["Mean r", "Median r", "SD", "Q1", "Q3", "Min", "Max"]} |
                        {c: "{:.2f}" for c in ["Fraction r>0.1", "Fraction r>0.3", "Fraction r>0.5"]})

    t3 = tests[["family", "comparison", "subset", "n_traits", "median_diff", "mean_diff",
                "ci_lo", "ci_hi", "p_wilcoxon", "frac_positive", "p_holm"]].copy()
    t3.columns = ["Family", "Comparison", "Subset", "n traits", "Median difference", "Mean difference",
                  "Bootstrap 2.5%", "Bootstrap 97.5%", "Wilcoxon P", "Fraction positive", "Holm P"]
    write(t3, "table3_paired_tests",
          caption="Table 3. Paired comparisons across traits (Wilcoxon signed-rank; Holm-corrected within the within-CV and portability families).",
          fmt={"Median difference": "{:.3f}", "Mean difference": "{:.3f}", "Bootstrap 2.5%": "{:.3f}",
               "Bootstrap 97.5%": "{:.3f}", "Wilcoxon P": "{:.3g}", "Holm P": "{:.3g}",
               "Fraction positive": "{:.2f}"})

    cols = ["trait_id", "group", "h2", "n", "n_expr",
            "kridge__G__random", "kridge__E__random", "kridge__GE__random",
            "kridge__G__lopo", "kridge__E__lopo", "kridge__GE__lopo",
            "lasso__G__random", "lasso__E__random", "lasso__GE__random",
            "lasso__G__lopo", "lasso__E__lopo", "lasso__GE__lopo",
            "hgb__G__random", "hgb__E__random", "hgb__GE__random",
            "hgb__G__lopo", "hgb__E__lopo", "hgb__GE__lopo"]
    tS1 = per[cols].copy()
    tS1.columns = ["Trait", "Domain", "h2", "n", "n matched"] + \
                  ["KRR G random", "KRR E random", "KRR G+E random",
                   "KRR G LOPO", "KRR E LOPO", "KRR G+E LOPO",
                   "Lasso G random", "Lasso E random", "Lasso G+E random",
                   "Lasso G LOPO", "Lasso E LOPO", "Lasso G+E LOPO",
                   "HGB G random", "HGB E random", "HGB G+E random",
                   "HGB G LOPO", "HGB E LOPO", "HGB G+E LOPO"]
    write(tS1, "tableS1_per_trait_accuracy", caption="Table S1. Per-trait out-of-fold Pearson r (mean of two random repeats).",
          fmt={c: "{:.3f}" for c in tS1.columns[5:]})

    p = os.path.join(ANA, "lopo_population_summary.csv")
    if os.path.exists(p):
        tS2 = pd.read_csv(p)
        tS2.columns = ["Features", "Held-out population", "n traits", "Median r", "Mean r"]
        write(tS2, "tableS2_lopo_populations", caption="Table S2. Within-population accuracy under leave-population-out CV.",
              fmt={"Median r": "{:.3f}", "Mean r": "{:.3f}"})

    p = os.path.join(RES, "transfer.csv")
    if os.path.exists(p):
        tr = pd.read_csv(p)
        tr = tr[["source", "target", "note", "model", "features", "n", "pheno_r", "r", "perm_p"]]
        tr.columns = ["Source", "Target", "Design", "Model", "Features", "n", "Phenotypic r", "Transfer r", "Permutation P"]
        write(tr, "tableS3_transfer", caption="Table S3. Cross-study and cross-environment transfer.",
              fmt={"Phenotypic r": "{:.3f}", "Transfer r": "{:.3f}", "Permutation P": "{:.3g}"})

    p = os.path.join(RES, "controls.csv")
    if os.path.exists(p):
        c = pd.read_csv(p)
        c = c[["trait", "features", "n", "real_r", "alpha", "w", "null_label_q95", "p_label",
               "null_kernel_q95", "p_kernel"]]
        c.columns = ["Trait", "Features", "n", "Observed r", "alpha", "Kernel weight", "Label-null 95%",
                     "Label P", "Kernel-null 95%", "Kernel P"]
        write(c, "tableS4_controls", caption="Table S4. Negative controls (300 label permutations; 100 kernel permutations).",
              fmt={"Observed r": "{:.3f}", "Label-null 95%": "{:.3f}", "Label P": "{:.3g}",
                   "Kernel-null 95%": "{:.3f}", "Kernel P": "{:.3g}"})

    p = os.path.join(RES, "enrichment.csv")
    if os.path.exists(p):
        e = pd.read_csv(p)
        e = e[["features", "score", "top_frac", "K", "floral_bg", "overlap", "p", "fold_enrich", "known_hits"]]
        e.columns = ["Features", "Ranking", "Top fraction", "Genes tested", "Floral background",
                     "Floral hits", "Hypergeometric P", "Fold enrichment", "Known flowering genes"]
        write(e, "tableS5_enrichment", caption="Table S5. Flowering-time gene-set enrichment among FT16-associated genes.",
              fmt={"Hypergeometric P": "{:.3g}", "Fold enrichment": "{:.2f}", "Top fraction": "{:.0%}"})


if __name__ == "__main__":
    main()
