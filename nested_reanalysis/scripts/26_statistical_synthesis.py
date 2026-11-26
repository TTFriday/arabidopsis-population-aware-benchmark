import hashlib
import json
import biosystems_lib as B
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = B.NEW / "results" / "synthesis"
SEED = 20261121
BOOTSTRAPS = 5000


def summarize(frame, label, metric):
    frame = frame.dropna(subset=["effect"]).copy()
    study_ids = sorted(frame.study_id.unique())
    blocks = [frame.loc[frame.study_id == s, "effect"].to_numpy() for s in study_ids]
    study_medians = np.array([np.median(block) for block in blocks])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(blocks), size=(BOOTSTRAPS, len(blocks)))
    pooled = np.array([np.median(np.concatenate([blocks[i] for i in draw])) for draw in draws])
    balanced = np.median(study_medians[draws], axis=1)
    row = {"comparison": label, "metric": metric, "n_units": len(frame), "n_traits": frame.trait_id.nunique(),
           "n_studies": len(blocks), "median_effect": frame.effect.median(),
           "block_ci_low": np.quantile(pooled, 0.025), "block_ci_high": np.quantile(pooled, 0.975),
           "study_balanced_median": np.median(study_medians),
           "balanced_ci_low": np.quantile(balanced, 0.025), "balanced_ci_high": np.quantile(balanced, 0.975),
           "positive_units": int((frame.effect > 0).sum()), "negative_units": int((frame.effect < 0).sum()),
           "positive_studies": int((study_medians > 0).sum()), "negative_studies": int((study_medians < 0).sum())}
    studies = frame.groupby("study_id", as_index=False).agg(n_units=("effect", "count"), n_traits=("trait_id", "nunique"),
                                                               median_effect=("effect", "median"), mean_effect=("effect", "mean"))
    studies.insert(0, "comparison", label)
    studies.insert(1, "metric", metric)
    sensitivity = []
    for omit in [[s] for s in study_ids] + [[12, 38]]:
        kept = frame[~frame.study_id.isin(omit)]
        sensitivity.append({"comparison": label, "metric": metric, "omitted_studies": "+".join(map(str, omit)),
                            "n_units": len(kept), "n_traits": kept.trait_id.nunique(),
                            "median_effect": kept.effect.median(),
                            "study_balanced_median": kept.groupby("study_id").effect.median().median()})
    return row, studies, pd.DataFrame(sensitivity)


def rank_association(frame, adjustment):
    x, y = rankdata(frame.fst), rankdata(frame.r)
    if adjustment != "none":
        design = pd.get_dummies(frame.trait_id, dtype=float).to_numpy()
        if adjustment in ("trait+size", "trait+size+spread"):
            design = np.column_stack([design, np.log(frame.n_test)])
        if adjustment == "trait+size+spread":
            design = np.column_stack([design, np.log(frame.relative_phenotype_sd)])
        x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if min(np.std(x), np.std(y)) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traits = pd.read_csv(B.NEW / "results/nested/per_trait.csv")
    sampling = pd.read_csv(B.NEW / "results/sampling/sampling_units.csv")
    summary_rows, study_tables, sensitivity_tables, effect_tables = [], [], [], []

    def add(frame, label, metric):
        row, studies, sensitivity = summarize(frame, label, metric)
        summary_rows.append(row)
        study_tables.append(studies)
        sensitivity_tables.append(sensitivity)
        effect_tables.append(frame.assign(comparison=label, metric=metric))

    for feature in ("G", "E", "GE"):
        for metric in ("r", "r2", "rmse"):
            wide = traits[traits.features == feature].pivot(index=["trait_id", "study_id"], columns="cv", values=metric)
            add((wide.random - wide.lopo).rename("effect").reset_index(), f"{feature}: random-minus-lopo", metric)
    for cv in ("random", "lopo"):
        wide = traits[traits.cv == cv].pivot(index=["trait_id", "study_id"], columns="features", values="r")
        for first, second in (("GE", "G"), ("G", "E"), ("G", "group_mean")):
            add((wide[first] - wide[second]).rename("effect").reset_index(), f"{cv}: {first}-minus-{second}", "r")
    for design in ("added", "size-matched"):
        data = sampling[sampling.design == design]
        for metric in ("r", "mean_abs_bias", "rmse", "r2"):
            wide = data.pivot(index=["trait_id", "study_id", "cluster6"], columns="fraction", values=metric)
            add((wide[0.5] - wide[0.0]).rename("effect").reset_index(), f"{design}: target-half-minus-zero", metric)
    half = sampling[sampling.fraction == 0.5].pivot(index=["trait_id", "study_id", "cluster6"], columns="design", values="r").dropna()
    add((half["size-matched"] - half.added).rename("effect").reset_index(), "eligible-56: size-matched-minus-added", "r")
    pd.DataFrame(summary_rows).to_csv(OUT / "paired_effects.csv", index=False)
    pd.concat(study_tables, ignore_index=True).to_csv(OUT / "study_effects.csv", index=False)
    pd.concat(sensitivity_tables, ignore_index=True).to_csv(OUT / "study_omission.csv", index=False)
    pd.concat(effect_tables, ignore_index=True).to_csv(OUT / "unit_effects.csv", index=False)

    decomposition = pd.read_csv(B.NEW / "results/nested/decomposition_raw.csv")
    decomposition = decomposition.groupby(["trait_id", "study_id", "features", "cv"], as_index=False).mean(numeric_only=True)
    decomposition.to_csv(OUT / "decomposition.csv", index=False)
    decomposition.groupby(["features", "cv"], as_index=False).agg(
        n_traits=("r_pooled", "count"), median_pooled=("r_pooled", "median"),
        median_group_centered=("r_group_centered", "median"), median_between_share=("phenotype_between_group_share", "median")
    ).to_csv(OUT / "decomposition_summary.csv", index=False)

    groups = pd.read_csv(B.NEW / "results/nested/groups_raw.csv")
    folds = pd.read_csv(B.NEW / "results/nested/folds_raw.csv")
    group_details = []
    for trait in traits.trait_id.unique():
        data = np.load(B.NEW / f"results/nested/{trait}__lopo__r0.npz")
        for group in np.unique(data["clusters"]):
            mask = data["clusters"] == group
            if mask.sum() < 15:
                continue
            fold = int(data["fold"][mask][0])
            fit = folds[(folds.trait_id == trait) & (folds.cv == "lopo") & (folds.features == "G") & (folds.fold == fold)].iloc[0]
            group_details.append({"trait_id": trait, "cluster6": group,
                                  "relative_phenotype_sd": np.std(data["y"][mask]) / np.std(data["y"]),
                                  "inner_type": fit.inner_type})
    groups = groups.merge(pd.DataFrame(group_details), on=["trait_id", "cluster6"], validate="many_to_one")
    groups.to_csv(OUT / "fst_units.csv", index=False)
    associations, omissions = [], []
    for feature in ("G", "E", "GE"):
        for subset in ("all", "grouped-inner-only"):
            data = groups[groups.features == feature]
            if subset != "all":
                data = data[data.inner_type == "grouped"]
            data = data.dropna(subset=["fst", "r", "relative_phenotype_sd"])
            data = data[data.relative_phenotype_sd > 0]
            blocks = [data[data.study_id == s] for s in sorted(data.study_id.unique())]
            draws = np.random.default_rng(SEED).integers(0, len(blocks), size=(2000, len(blocks)))
            for adjustment in ("none", "trait", "trait+size", "trait+size+spread"):
                bootstrap = np.array([rank_association(pd.concat([blocks[i] for i in draw], ignore_index=True), adjustment) for draw in draws])
                associations.append({"features": feature, "subset": subset, "adjustment": adjustment,
                                     "n_units": len(data), "n_traits": data.trait_id.nunique(), "n_studies": len(blocks),
                                     "rho": rank_association(data, adjustment),
                                     "block_ci_low": np.nanquantile(bootstrap, 0.025),
                                     "block_ci_high": np.nanquantile(bootstrap, 0.975),
                                     "finite_bootstraps": int(np.isfinite(bootstrap).sum())})
                for omit in [[s] for s in sorted(data.study_id.unique())] + [[12, 38]]:
                    kept = data[~data.study_id.isin(omit)]
                    omissions.append({"features": feature, "subset": subset, "adjustment": adjustment,
                                      "omitted_studies": "+".join(map(str, omit)), "n_units": len(kept),
                                      "rho": rank_association(kept, adjustment)})
    pd.DataFrame(associations).to_csv(OUT / "fst_associations.csv", index=False)
    pd.DataFrame(omissions).to_csv(OUT / "fst_study_omission.csv", index=False)
    distinct = folds.drop_duplicates(["trait_id", "cv", "repeat", "fold"])
    distinct.groupby(["cv", "inner_type"], as_index=False).size().to_csv(OUT / "tuning_counts.csv", index=False)
    hashes = {}
    for name in ("biosystems_lib.py", "24_nested_validation.py", "25_nested_sampling.py", "26_statistical_synthesis.py", "test_nested_validation.py"):
        path = B.NEW / "scripts" / name
        hashes[str(path.relative_to(B.NEW))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [B.PROC / n for n in ("geno_matrix.npy", "accessions.csv", "expr_accessions.csv", "phenotypes_long.csv", "pop6.csv", "PCs.csv", "benchmark_traits.csv")] + [B.EXPR]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    (OUT / "analysis_record.json").write_text(json.dumps({
        "seed": SEED, "effect_block_bootstraps": BOOTSTRAPS, "fst_block_bootstraps": 2000,
        "block": "Source study; all traits or trait-group units retained together",
        "effect_statistic": "Median unit effect; study-balanced statistic is median of within-study median effects",
        "fst_statistic": "Pearson correlation of ranks, or residual ranks after trait indicators and stated log covariates",
        "uncertainty_scope": "Exploratory empirical study-block stability intervals, not a population confidence guarantee; only eight source studies and overlapping accessions",
        "analysis_timing": "Retrospective reanalysis designed after examination of archived analyses; not preregistered",
        "sha256": hashes
    }, indent=2), encoding="utf8")
    print(pd.DataFrame(summary_rows).query("metric == 'r'").to_string(index=False))
    print(pd.DataFrame(associations).query("features == 'G'").to_string(index=False))


if __name__ == "__main__":
    main()
