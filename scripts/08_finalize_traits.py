"""Finalize the benchmark trait panel; excludes >30% zero-inflated traits."""
import os
import pandas as pd

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(OUT, "data_processed")

sel = pd.read_csv(os.path.join(OUT, "results", "selected_traits.csv"))
sel["key"] = sel.study_id.astype(str) + "|" + sel.trait

picks = {
    12: ["FT10", "FT16"],
    38: ["DTF2", "CL", "RL", "RBN"],
    31: ["rosetteDM", "ScalingExponent", "FruitNumber", "RGR"],
    43: ["stomata density 261", "delta 13C 261", "Stomatasize 261", "stomata density", "stomatasize"],
    126: ["Trichome density", "Trichome stem length", "Trichome area",
          "Trichome autofluorescence", "Number of trichome branches", "Trichome length"],
    34: ["rSeeds thp", "rSeeds thi", "rFitness thp", "rFitness mhi", "rSurvival fruit mli", "rSeeds mhp"],
    127: ["Raffinose 22C", "Raffinose 32C"],
    37: ["Inter-specific pollination of Arabidopsis thaliana and Malcolmia littorea"],
}
keys = [f"{sid}|{tr}" for sid, trs in picks.items() for tr in trs]
sel = sel[sel.key.isin(keys)].copy()

labels = {
    "FT10": "FT10", "FT16": "FT16", "DTF2": "DTF2", "CL": "CL", "RL": "RL", "RBN": "RBN",
    "rosetteDM": "rosetteDM", "ScalingExponent": "scalingExp", "FruitNumber": "fruitNo", "RGR": "RGR",
    "stomata density 261": "stomataDens_A", "delta 13C 261": "delta13C_A", "Stomatasize 261": "stomataSize_A",
    "stomata density": "stomataDens_B", "stomatasize": "stomataSize_B",
    "Metals in stomata": "metalStomata", "Leaf epidermis autofluorescence": "leafAutofluor",
    "Trichome density": "trichDensity", "Trichome stem length": "trichStemLen",
    "Trichome area": "trichArea", "Trichome autofluorescence": "trichAutofluor",
    "Number of trichome branches": "trichBranches", "Trichome length": "trichLen",
    "rSeeds thp": "seeds_thp", "rSeeds thi": "seeds_thi", "rFitness thp": "fit_thp",
    "rFitness mhi": "fit_mhi", "rSurvival fruit mli": "surv_mli", "rSeeds mhp": "seeds_mhp",
    "Raffinose 22C": "raffinose22", "Raffinose 32C": "raffinose32",
    "Inter-specific pollination of Arabidopsis thaliana and Malcolmia littorea": "pollination",
}
groups = {12: "flowering", 38: "flowering/architecture", 31: "growth allometry",
          43: "stomata/gas exchange", 126: "trichome/metal", 34: "fitness",
          127: "seed chemistry", 37: "pollination"}
sel["trait_id"] = sel.trait.map(labels)
sel["group"] = sel.study_id.map(groups)
sel = sel.sort_values(["group", "study_id", "h2"], ascending=[True, True, False])
sel.to_csv(os.path.join(PROC, "benchmark_traits.csv"), index=False)
print(sel[["study_id", "trait_id", "group", "n", "n_expr", "h2"]].to_string(index=False))
print("\ntraits", len(sel), "studies", sel.study_id.nunique())
