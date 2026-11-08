"""Known-biology check: flowering-time gene enrichment among top G/E features."""
import os, sys, gzip, re
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.linear_model import LassoCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import Panel, OUT, PROC

DATA = os.environ.get("ARABIDOPSIS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
GFF = os.path.join(DATA, "06_genome_TAIR10/Arabidopsis_thaliana.TAIR10.63.gff3.gz")
GAF = os.path.join(DATA, "05_annotation_GO_KEGG/tair.gaf.gz")
OBO = os.path.join(DATA, "05_annotation_GO_KEGG/go-basic.obo")

TARGET_GO = {
    "GO:0009908": "flower development",
    "GO:0048573": "photoperiodism, flowering",
    "GO:0010228": "vegetative to reproductive phase transition",
    "GO:0009910": "negative regulation of flower development",
    "GO:0009911": "positive regulation of flower development",
    "GO:0048574": "long-day photoperiodism, flowering",
}
KNOWN = {
    "AT4G00650": "FRI", "AT5G10140": "FLC", "AT1G65480": "FT", "AT2G45660": "SOC1",
    "AT1G77080": "MAF1/FLM", "AT5G65050": "MAF2", "AT5G65060": "MAF3", "AT5G65070": "MAF4",
    "AT5G65080": "MAF5", "AT5G57380": "VIN3", "AT3G18990": "VRN1", "AT4G16845": "VRN2",
    "AT1G22770": "GI", "AT5G15840": "CO", "AT4G08920": "CRY1", "AT1G04400": "CRY2",
    "AT1G09570": "PHYA", "AT2G18790": "PHYB", "AT5G35840": "PHYC", "AT4G16250": "PHYD",
    "AT4G18130": "PHYE", "AT2G22540": "SVP", "AT4G24540": "AGL24", "AT5G61850": "LFY",
    "AT4G36920": "AP1", "AT4G20370": "TSF", "AT1G25560": "TEM1", "AT1G68840": "TEM2",
    "AT4G22140": "EBS", "AT3G12810": "PIE1", "AT5G23150": "HUA2", "AT2G25930": "ELF3",
    "AT2G40080": "ELF4", "AT2G40970": "LUX", "AT5G02810": "PRR7", "AT2G46790": "PRR9",
    "AT5G61380": "TOC1", "AT5G57360": "ZTL", "AT1G01060": "LHY", "AT2G46830": "CCA1",
}


def parse_genes():
    genes = {}
    with gzip.open(GFF, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene": continue
            attrs = dict(x.split("=", 1) for x in p[8].split(";") if "=" in x)
            gid = attrs.get("ID", "").replace("gene:", "")
            if gid.startswith("AT"):
                genes[gid] = (p[0], int(p[3]), int(p[4]))
    return genes


def parse_go():
    gene2go = {}
    with gzip.open(GAF, "rt") as f:
        for line in f:
            if line.startswith("!"): continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9: continue
            gid = p[1].split(".")[0]
            if not gid.startswith("AT"): continue
            if "NOT" in p[3]: continue
            gene2go.setdefault(gid, set()).add(p[4])
    # descendants
    parents = {}
    cur = None
    with open(OBO, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]": cur = None
            elif line.startswith("id: GO:"): cur = line[4:]
            elif cur and line.startswith("is_a: GO:"):
                parents.setdefault(line[6:].split()[0], set()).add(cur)
            elif cur and line.startswith("relationship: part_of GO:"):
                parents.setdefault(line.split()[2], set()).add(cur)
    desc = {}
    for t in TARGET_GO:
        seen, stack = set(), [t]
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x); stack.extend(parents.get(x, ()))
        desc[t] = seen
    floral = set()
    for t, ds in desc.items():
        for g, gos in gene2go.items():
            if gos & ds: floral.add(g)
    return gene2go, floral, desc


def map_snps(genes, meta):
    bychr = {}
    for g, (c, s, e) in genes.items():
        bychr.setdefault(str(c), []).append((s, e, g))
    for c in bychr: bychr[c].sort()
    starts_by_chr = {c: [x[0] for x in arr] for c, arr in bychr.items()}
    out = []
    import bisect
    for r in meta.itertuples():
        arr = bychr.get(str(r.chr), [])
        if not arr: out.append(None); continue
        starts = starts_by_chr[str(r.chr)]
        i = bisect.bisect_right(starts, r.pos) - 1
        hit = None
        for j in range(max(0, i - 2), min(len(arr), i + 3)):
            s, e, g = arr[j]
            if s <= r.pos <= e: hit = g; break
        out.append(hit)
    return out


def enrich(top, background, floral):
    top, background = set(top), set(background)
    M, K = len(background), len(top)
    n = len(background & floral); k = len(top & floral)
    p = hypergeom.sf(k - 1, M, n, K) if K and n else 1.0
    fe = (k / K) / (n / M) if K and n and M else np.nan
    return k, n, K, M, p, fe


def main():
    genes = parse_genes(); gene2go, floral, desc = parse_go()
    print("genes", len(genes), "annotated", len(gene2go), "floral", len(floral), flush=True)
    meta = pd.read_csv(os.path.join(PROC, "geno_meta.csv"))
    meta["gene"] = map_snps(genes, meta)
    print("SNPs mapped", meta.gene.notna().mean().round(3), flush=True)

    P = Panel(); ids, y, cl = P.trait(12, "FT16")
    ig = [P.gidx[a] for a in ids]; ie = [P.eidx[a] for a in ids]
    Xg = P.G[ig]; Xe = P.E[ie]
    # map panel SNP rows to full-matrix rows through genotype order: geno_matrix rows correspond to acc order? use SNP identity by position
    gene_ids = pd.read_csv(os.path.join(PROC, "expr_genes.csv")).gene_id.values
    rows = []
    # univariate + lasso for G
    for name, X, bg_genes in [("G", Xg, meta.gene.dropna().unique()), ("E", Xe, gene_ids)]:
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        r = np.abs(Xs.T @ y) / len(y)
        lasso = LassoCV(cv=5, alphas=30, eps=1e-3, max_iter=3000, tol=1e-3, random_state=0)
        lasso.fit(Xs, y)
        coef = np.abs(lasso.coef_)
        if name == "G":
            gene_r = meta.assign(r=r).groupby("gene")["r"].max()
            gene_c = meta.assign(c=coef).groupby("gene")["c"].max()
            # align: meta row order == Xg columns order (both from geno_matrix)
            gene_r = gene_r.dropna()
            gene_c = gene_c.dropna()
        else:
            gene_r = pd.Series(r, index=gene_ids)
            gene_c = pd.Series(coef, index=gene_ids)
        for score_name, score in [("univariate", gene_r), ("lasso", gene_c)]:
            score = score.loc[score.index.isin(genes)].sort_values(ascending=False)
            bg = list(score.index)
            for frac in (0.02, 0.05, 0.10):
                K = max(10, int(len(bg) * frac))
                top = list(score.index[:K])
                k, n, kk, M, p, fe = enrich(top, bg, floral)
                hit = [KNOWN[g] for g in top if g in KNOWN]
                rows.append({"features": name, "score": score_name, "top_frac": frac,
                             "K": kk, "floral_bg": n, "overlap": k, "p": p, "fold_enrich": fe,
                             "known_hits": ";".join(sorted(set(hit))), "alpha": lasso.alpha_ if score_name == "lasso" else np.nan})
                print(f"{name} {score_name} top{frac:.0%}: floral {k}/{kk} bg {n}/{M} p={p:.2e} FE={fe:.2f} known={hit}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "results", "enrichment.csv"), index=False)
    meta.to_csv(os.path.join(PROC, "snp_gene_map.csv"), index=False)


if __name__ == "__main__":
    main()
