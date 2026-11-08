"""G3-standard figures (vector PDF + 300 dpi PNG)."""
import os, sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, wilcoxon
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_lib import Panel, OUT, PROC

RES = os.path.join(OUT, "results")
ANA = os.path.join(RES, "analysis")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.8, "axes.titlesize": 10.4,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 1.2,
})

C_G, C_E, C_GE = "#0072B2", "#D55E00", "#009E73"
C_FEAT = {"G": C_G, "E": C_E, "GE": C_GE}
C_MODEL = {"kridge": "#0072B2", "lasso": "#D55E00", "hgb": "#CC79A7"}
D_MODEL = {"kridge": "KRR", "lasso": "Lasso", "hgb": "HGB"}
D_FEAT = {"G": "G", "E": "E", "GE": "G+E"}
D_CV = {"random": "Random", "lopo": "LOPO"}
GREY = "#6E6E6E"
GROUP_ORDER = ["fitness", "flowering", "flowering/architecture", "growth allometry",
               "pollination", "seed chemistry", "stomata/gas exchange", "trichome/metal"]
C_GROUP = {"fitness": "#E69F00", "flowering": "#56B4E9",
           "flowering/architecture": "#009E73", "growth allometry": "#8C6D31",
           "pollination": "#0072B2", "seed chemistry": "#D55E00",
           "stomata/gas exchange": "#CC79A7", "trichome/metal": "#8A8A8A"}
NICE = {"fitness": "Fitness", "flowering": "Flowering", "flowering/architecture": "Flowering/architecture",
        "growth allometry": "Growth allometry", "pollination": "Pollination",
        "seed chemistry": "Seed chemistry", "stomata/gas exchange": "Stomata/gas exchange",
        "trichome/metal": "Trichome/metal"}


def panel_label(ax, s, dx=-0.16, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11.7, fontweight="bold",
            va="top", ha="left")


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def load_analysis():
    per = pd.read_csv(os.path.join(ANA, "per_trait.csv"))
    tests = pd.read_csv(os.path.join(ANA, "tests.csv"))
    summ = pd.read_csv(os.path.join(ANA, "summary_by_combo.csv"))
    return per, tests, summ


def trait_order(per):
    per = per.copy()
    per["_g"] = pd.Categorical(per.group, GROUP_ORDER, ordered=True)
    per = per.sort_values(["_g", "h2"], ascending=[True, False]).reset_index(drop=True)
    return per


def ptext(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return f"P={p:.1e}"
    return f"P={p:.3f}"


def get_test(tests, comp, subset=None):
    t = tests[tests.comparison == comp]
    if subset is not None:
        t = t[t.subset == subset]
    return t.iloc[0] if len(t) else None


def heat_block(ax, M, row_labels, col_labels, norm, cmap, xlabels=True, label_fs=5.2):
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.8)
    ax.set_xticks(np.arange(len(col_labels)))
    if xlabels:
        ax.set_xticklabels(col_labels, rotation=90, fontsize=label_fs)
    else:
        ax.set_xticklabels([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def load_panels():
    P = Panel()
    pop = pd.read_csv(os.path.join(PROC, "pop_clusters.csv"), dtype={"accession_id": str})
    pc = pd.read_csv(os.path.join(PROC, "PCs.csv"), dtype={"accession_id": str})
    df = pop.merge(pc[["accession_id", "PC1", "PC2"]], on="accession_id")
    return P, df


def pop_short_label(g):
    shares = g.country.fillna("NA").value_counts(normalize=True)
    names = {"GER": "GER", "Czech Republic": "CZE", "Spain": "ESP", "Sweden": "SWE",
             "Russia": "RUS", "Armenia": "ARM", "Finland": "FIN", "UK": "UK",
             "US": "US", "Italy": "ITA", "France": "FRA", "Kyrgyzstan": "KGZ",
             "Uzbekistan": "UZB", "Azerbaijan": "AZE", "Greece": "GRC"}
    top = [names.get(k, k) for k in shares.head(2).index]
    return "/".join(top)


# ---------------------------------------------------------------- Figure 1
def fig1(per):
    P, df = load_panels()
    v = eigh(np.load(os.path.join(PROC, "GRM.npy")), eigvals_only=True)[::-1]
    pv = v / v.sum() * 100

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.55], hspace=0.28, wspace=0.42,
                          left=0.07, right=0.98, top=0.96, bottom=0.08)

    # A: data flow
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_label(ax, "A", dx=-0.02, dy=1.10)

    def box(x, y, w, h, title, body, color, tfs=6.4, bfs=5.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.015",
                                    fc=color, ec="none", alpha=0.18))
        ax.text(x + w/2, y + h*0.72, title, ha="center", va="center", fontsize=tfs, fontweight="bold")
        ax.text(x + w/2, y + h*0.32, body, ha="center", va="center", fontsize=bfs, color="#222222")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.6, color=GREY, shrinkA=2, shrinkB=2))

    for x, s in [(0.105, "Raw data"), (0.385, "Processed features"),
                 (0.665, "Aligned panel"), (0.905, "Evaluation")]:
        ax.text(x, 0.98, s, ha="center", va="top", fontsize=7.5, color=GREY, style="italic")

    box(0.005, 0.63, 0.20, 0.24, "Genotypes", "1,135 accessions\n10,709,949 SNPs", C_G)
    box(0.005, 0.35, 0.20, 0.24, "Transcriptomes", "727 accessions\n23,008 expressed genes", C_E)
    box(0.005, 0.07, 0.20, 0.24, "Phenotypes", "30 studies | 536 traits\nAraPheno", C_GE)

    box(0.285, 0.63, 0.20, 0.24, "", "12,846 LD-pruned SNPs\nGRM (1,135$^2$) + 50 PCs", C_G, bfs=5.5)
    box(0.285, 0.35, 0.20, 0.24, "", "23,008-gene expression\nkernel (727$^2$) + 50 PCs", C_E, bfs=5.5)
    box(0.285, 0.07, 0.20, 0.24, "", "27 traits | 8 studies\n8 trait domains", C_GE, bfs=5.5)

    box(0.565, 0.10, 0.19, 0.74, "Population-aware\npanel",
        "654-accession union\nG + E + ≥1 phenotype\n\n6 initial clusters\n2–6 groups per trait\n\n27 traits", "none", tfs=5.8, bfs=5.4)
    ax.add_patch(FancyBboxPatch((0.562, 0.10), 0.185, 0.74, boxstyle="round,pad=0.010,rounding_size=0.015",
                                fc="none", ec="#555555", lw=0.6))

    box(0.815, 0.10, 0.18, 0.74, "Prediction\nbenchmark",
        "KRR\nLasso\nHGB\n\nrandom 5-fold\nLOPO\n\npermutation\ncontrols\ntransfer", "none", tfs=5.8, bfs=5.2)
    ax.add_patch(FancyBboxPatch((0.805, 0.10), 0.175, 0.74, boxstyle="round,pad=0.010,rounding_size=0.015",
                                fc="none", ec="#555555", lw=0.6))

    for y in [0.75, 0.47, 0.19]:
        arrow(0.208, y, 0.282, y)
        arrow(0.488, y, 0.560, 0.47)
    arrow(0.752, 0.47, 0.802, 0.47)

    # B: PCA
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "B")
    for c in sorted(df.cluster6.unique()):
        g = df[df.cluster6 == c]
        ax.scatter(g.PC1, g.PC2, s=4.5, lw=0, color=list(C_GROUP.values())[int(c) % 8],
                   alpha=0.8, label=pop_short_label(g))
    ax.set_xlabel(f"PC1 ({pv[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({pv[1]:.1f}%)")
    ax.legend(markerscale=2.2, fontsize=6.8, loc="lower right", handletextpad=0.2, borderpad=0.1)
    ax.set_title("Genetic structure", pad=3)

    # C: geography
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "C")
    geo = df[df.lon.notna() & df.lat.notna()]
    for c in sorted(df.cluster6.unique()):
        g = geo[geo.cluster6 == c]
        ax.scatter(g.lon, g.lat, s=4.5, lw=0, color=list(C_GROUP.values())[int(c) % 8], alpha=0.75)
    ax.set_xlim(-11, 45); ax.set_ylim(34, 72)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Collection geography", pad=3)
    axin = ax.inset_axes([0.68, 0.04, 0.30, 0.24])
    us = geo[geo.country == "US"]
    axin.scatter(us.lon, us.lat, s=3, lw=0, color=list(C_GROUP.values())[3], alpha=0.8)
    axin.set_xlim(-126, -65); axin.set_ylim(24, 50)
    axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_linewidth(0.4); s.set_color(GREY)
    axin.text(0.5, 0.98, "US (n=97)", transform=axin.transAxes, ha="center", va="top", fontsize=6.5)
    ax.text(0.98, 0.97, f"n = {len(geo):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color=GREY)

    # D: trait panel
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "D")
    p = trait_order(per)
    y = np.arange(len(p))
    ax.hlines(y, 0, p.h2, color=[C_GROUP[g] for g in p.group], lw=0.8, alpha=0.55)
    ax.scatter(p.h2, y, s=0.07 * p.n_expr, color=[C_GROUP[g] for g in p.group],
               edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(p.trait_id, fontsize=6.1)
    ax.set_xlabel("REML heritability (h$^2$)")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.8, len(p) - 0.2)
    ax.invert_yaxis()
    ax.set_title("27-trait benchmark panel", pad=3)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=C_GROUP[g],
                      markersize=3.2, label=NICE[g]) for g in GROUP_ORDER]
    ax.legend(handles=handles, fontsize=6.6, loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=2,
              handletextpad=0.15, columnspacing=0.6, borderpad=0.1, labelspacing=0.35, frameon=False)

    fig.text(0.01, 0.005, "Marker area in D is proportional to the number of accessions with matched transcriptome; n in C counts accessions with coordinates.",
             fontsize=6.5, color=GREY)
    save(fig, "fig1_design")


# ---------------------------------------------------------------- Figure 2
def fig2(per, tests):
    p = trait_order(per)
    rows = [("kridge", "G"), ("kridge", "E"), ("kridge", "GE"),
            ("lasso", "G"), ("lasso", "E"), ("lasso", "GE"),
            ("hgb", "G"), ("hgb", "E"), ("hgb", "GE")]
    row_labels = []
    for m, f in rows:
        row_labels.append(f"{D_MODEL[m]} {D_FEAT[f]}")
    Mrand = np.vstack([p[f"{m}__{f}__random"].values for m, f in rows])
    Mlop = np.vstack([p[f"{m}__{f}__lopo"].values for m, f in rows])
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.9)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.46, wspace=0.38,
                          left=0.11, right=0.97, top=0.91, bottom=0.155)
    fig.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=C_GROUP[g],
                           markersize=4, label=NICE[g]) for g in GROUP_ORDER],
               fontsize=6.8, loc="upper left", bbox_to_anchor=(0.10, 0.995), ncol=4,
               handletextpad=0.2, columnspacing=1.0, labelspacing=0.45, frameon=False)

    ax1 = fig.add_subplot(gs[0, 0:2])
    im = heat_block(ax1, Mrand, row_labels, p.trait_id, norm, "RdBu_r")
    panel_label(ax1, "A", dx=-0.20, dy=1.08)
    pos1 = ax1.get_position()
    axs = fig.add_axes([pos1.x0, pos1.y1 + 0.007, pos1.width, 0.011])
    strip = np.array([[list(C_GROUP.values()).index(C_GROUP[g]) for g in p.group]])
    cmap = ListedColormap([C_GROUP[g] for g in GROUP_ORDER])
    axs.imshow(strip, aspect="auto", cmap=cmap, vmin=0, vmax=len(GROUP_ORDER)-1)
    axs.set_xticks([]); axs.set_yticks([])
    for s in axs.spines.values():
        s.set_visible(False)
    ax2 = fig.add_subplot(gs[1, 0:2])
    heat_block(ax2, Mlop, row_labels, p.trait_id, norm, "RdBu_r")
    for _ax, _lab in [(ax1, "Random 5-fold"), (ax2, "Leave-population-out")]:
        _pos = _ax.get_position()
        fig.text(0.004, (_pos.y0 + _pos.y1) / 2, _lab, rotation=90, va="center",
                 ha="center", fontsize=8.5)

    cax = fig.add_axes([0.305, 0.050, 0.17, 0.009])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Pearson r (mean across repeats)", fontsize=7.2, labelpad=1)
    cb.ax.tick_params(labelsize=5.2, length=2)
    cb.outline.set_linewidth(0.4)

    # B: slopegraph
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "B", dx=-0.20)
    for f, off in [("G", -0.06), ("GE", 0.06)]:
        a = p[f"kridge__{f}__random"].values; b = p[f"kridge__{f}__lopo"].values
        for i in range(len(p)):
            ax.plot([0 + off, 1 + off], [a[i], b[i]], color=C_FEAT[f], lw=0.4, alpha=0.28)
        ax.plot([0 + off, 1 + off], [np.nanmedian(a), np.nanmedian(b)],
                color=C_FEAT[f], lw=1.8, marker="o", ms=3, label=D_FEAT[f])
    ax.set_xlim(-0.25, 1.25); ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random", "LOPO"])
    ax.set_ylabel("Pearson r"); ax.set_ylim(-0.25, 0.85)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("Portability drop", pad=3)

    # C: delta r
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "C", dx=-0.20)
    data, cols, labs = [], [], []
    for cv in ["random", "lopo"]:
        data.append((p[f"kridge__GE__{cv}"] - p[f"kridge__G__{cv}"]).dropna().values)
        cols.append("#4D4D4D"); labs.append(cv.upper() if cv == "lopo" else "Random")
    vp = ax.violinplot(data, positions=[0, 1], widths=0.6, showextrema=False)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(cols[i]); b.set_alpha(0.25); b.set_edgecolor("none")
    for i, d in enumerate(data):
        x = i + np.random.default_rng(7).normal(0, 0.035, len(d))
        ax.scatter(x, d, s=5, color=cols[i], alpha=0.7, lw=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(d)] * 2, color="black", lw=1.2)
    ax.axhline(0, color=GREY, lw=0.5, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labs)
    ax.set_ylabel("$\\Delta$r (G+E $-$ G)")
    t = get_test(tests, "GE_vs_G (kridge)", "random")
    t2 = get_test(tests, "GE_vs_G (kridge)", "lopo")
    ax.set_title(f"Expression gain\nrandom {ptext(t.p_wilcoxon)} | LOPO {ptext(t2.p_wilcoxon)}",
                 fontsize=8.1, pad=3)

    fig.text(0.01, 0.005, "A: repeat means; B/C summaries: trait medians. LOPO, leave-population-out; r, out-of-fold Pearson correlation.",
             fontsize=6.5, color=GREY)
    save(fig, "fig2_accuracy")


# ---------------------------------------------------------------- Figure 3
def fig3(per, tests):
    p = trait_order(per)
    gapG = (p["kridge__G__random"] - p["kridge__G__lopo"]).values
    gapGE = (p["kridge__GE__random"] - p["kridge__GE__lopo"]).values
    order = np.argsort(gapGE)
    y = np.arange(len(p))

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.20, 0.95], hspace=0.50, wspace=0.50,
                          left=0.13, right=0.98, top=0.94, bottom=0.10)

    # A: paired gaps
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A", dx=-0.42)
    ax.barh(y - 0.18, gapG[order], height=0.34, color=C_G, alpha=0.85, lw=0, label="G")
    ax.barh(y + 0.18, gapGE[order], height=0.34, color=C_GE, alpha=0.85, lw=0, label="G+E")
    ax.set_yticks(y); ax.set_yticklabels(p.trait_id.values[order], fontsize=6.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Portability gap (random $-$ LOPO, r)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.invert_yaxis()
    ax.set_title("Accuracy lost under LOPO", fontsize=8.8, pad=3)

    # B: LOPO within-population heatmap
    bypop = pd.read_csv(os.path.join(ANA, "lopo_by_population.csv"))
    psum = bypop.groupby(["features", "population"]).r.median().unstack(0)
    pop_order = (bypop.groupby("population").n_pop.median().sort_values(ascending=False).index)
    psum = psum.loc[pop_order]
    M = psum[["G", "E", "GE"]].values
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B", dx=-0.34)
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.7)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["G", "E", "G+E"])
    ax.set_yticks(np.arange(len(psum)))
    ax.set_yticklabels([s.split(" (")[0] for s in psum.index], fontsize=6.8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.45 or M[i, j] < -0.18 else "black")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Within-group accuracy (LOPO)", fontsize=8.8, pad=3)

    # C: expression gain within populations
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "C", dx=-0.34)
    d = bypop.pivot_table(index=["trait_id", "population"], columns="features", values="r")
    d = d.dropna(subset=["G", "GE"])
    d["delta"] = d["GE"] - d["G"]
    grp = d.groupby("population").delta
    med = pd.DataFrame({"mean": grp.mean(), "n": grp.size()}).loc[pop_order]
    rng = np.random.default_rng(0)
    lo, hi = [], []
    for pop in med.index:
        arr = grp.get_group(pop).values
        bs = rng.choice(arr, (4000, len(arr)), replace=True).mean(1)
        lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    ypos = np.arange(len(med))
    ax.barh(ypos, med["mean"].values, color=C_GE, alpha=0.85, lw=0)
    ax.errorbar(med["mean"].values, ypos, xerr=[med["mean"].values - np.array(lo),
                np.array(hi) - med["mean"].values], fmt="none", ecolor="black",
                elinewidth=0.7, capsize=2)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{s.split(' (')[0]} ({int(n)})" for s, n in zip(med.index, med["n"])],
                       fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean $\\Delta$r (G+E $-$ G)")
    ax.set_title("Expression gain by held-out group", fontsize=8.2, pad=3)

    # D: gap vs h2
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "D", dx=-0.42)
    for g in GROUP_ORDER:
        m = p.group.values == g
        ax.scatter(p.h2[m], gapGE[m], s=16, color=C_GROUP[g], edgecolor="white",
                   linewidth=0.3, alpha=0.9, label=NICE[g])
    rho, pv = spearmanr(p.h2, gapGE, nan_policy="omit")
    mfit = np.isfinite(p.h2.values) & np.isfinite(gapGE)
    xs = np.linspace(np.nanmin(p.h2), np.nanmax(p.h2), 20)
    if mfit.sum() > 2:
        b, a = np.polyfit(p.h2.values[mfit], gapGE[mfit], 1)
        ax.plot(xs, a + b * xs, color="black", lw=0.8, ls="--")
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xlabel("REML heritability (h$^2$)")
    ax.set_ylabel("Portability gap, G+E (r)")
    ax.set_title("Gap vs heritability", fontsize=8.5, pad=3)
    ax.text(0.03, 0.05, f"$\\rho$={rho:.2f}, {ptext(pv)}", transform=ax.transAxes, fontsize=6.8)

    # E: accuracy vs h2, random vs LOPO
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "E", dx=-0.42)
    for cv, mk, ls in [("random", "o", "-"), ("lopo", "^", "--")]:
        rr = p[f"kridge__GE__{cv}"].values
        ax.scatter(p.h2, rr, s=13, marker=mk,
                   color=C_GE if cv == "random" else "white", edgecolor=C_GE,
                   linewidth=0.6, alpha=0.9 if cv == "random" else 1)
        rho, pv = spearmanr(p.h2, rr, nan_policy="omit")
        mfit = np.isfinite(p.h2.values) & np.isfinite(rr)
        if mfit.sum() > 2:
            b, a = np.polyfit(p.h2.values[mfit], rr[mfit], 1)
            ax.plot(xs, a + b * xs, color=C_GE, lw=0.8, ls=ls)
        ax.text(0.03, 0.95 if cv == "random" else 0.86,
                f"{cv}: $\\rho$={rho:.2f}, {ptext(pv)}", transform=ax.transAxes,
                fontsize=6.8, color=C_GE)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xlabel("REML heritability (h$^2$)")
    ax.set_ylabel("Pearson r (KRR G+E)")
    ax.set_title("Accuracy vs heritability", fontsize=8.5, pad=3)
    accuracy = p[["kridge__GE__random", "kridge__GE__lopo"]].to_numpy().ravel()
    accuracy = accuracy[np.isfinite(accuracy)]
    if len(accuracy):
        margin = max(0.03, 0.05 * np.ptp(accuracy))
        ax.set_ylim(accuracy.min() - margin, accuracy.max() + margin)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C_GE, lw=0, ms=3, label="Random"),
                       Line2D([0], [0], marker="^", color=C_GE, lw=0, ms=3,
                              markerfacecolor="white", label="LOPO")],
              fontsize=6.8, loc="lower right")

    # F: best feature under LOPO
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "F", dx=-0.42)
    best = pd.read_csv(os.path.join(ANA, "best_feature_per_trait.csv"))
    cnt = best.best_lopo.value_counts().reindex(["G", "E", "GE"]).fillna(0)
    cnt2 = best.best_random.value_counts().reindex(["G", "E", "GE"]).fillna(0)
    x = np.arange(3)
    ax.bar(x - 0.18, cnt2.values, width=0.34, color="#BDBDBD", label="Random")
    ax.bar(x + 0.18, cnt.values, width=0.34,
           color=[C_G, C_E, C_GE], label="LOPO")
    for i, (a, b) in enumerate(zip(cnt2.values, cnt.values)):
        ax.text(i - 0.18, a + 0.4, int(a), ha="center", fontsize=7.2)
        ax.text(i + 0.18, b + 0.4, int(b), ha="center", fontsize=7.2)
    ax.set_xticks(x); ax.set_xticklabels(["G", "E", "G+E"])
    ax.set_ylabel("Number of traits")
    ax.set_ylim(0, max(cnt.max(), cnt2.max()) * 1.25)
    ax.set_yticks(np.arange(0, max(cnt.max(), cnt2.max()) * 1.25 + 1, 5))
    ax.legend(fontsize=7)
    ax.set_title("Best feature set", fontsize=8.5, pad=3)

    fig.text(0.01, 0.005, "B/C: trait-specific merged groups; ≥15 observations per trait/group. Parentheses in C give trait counts.",
             fontsize=6.5, color=GREY)
    save(fig, "fig3_portability")


# ---------------------------------------------------------------- Figure 4
def fig4(per, tests):
    p = trait_order(per)
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                          left=0.09, right=0.98, top=0.95, bottom=0.13)

    for k, (cv, m2, xlab) in enumerate([("random", "lasso", "a"), ("lopo", "hgb", "b")]):
        ax = fig.add_subplot(gs[0, k])
        panel_label(ax, "AB"[k], dx=-0.18)
        x = p[f"kridge__GE__{cv}"].values
        y = p[f"{m2}__GE__{cv}"].values
        for g in GROUP_ORDER:
            m = p.group.values == g
            ax.scatter(x[m], y[m], s=15, color=C_GROUP[g], edgecolor="white", lw=0.3, label=NICE[g])
        ax.plot([-0.2, 0.9], [-0.2, 0.9], color=GREY, lw=0.6, ls="--")
        ax.axhline(0, color=GREY, lw=0.4); ax.axvline(0, color=GREY, lw=0.4)
        comp = f"kridge_vs_{m2} (GE)"
        t = get_test(tests, comp, cv)
        cvl = "Random" if cv == "random" else "LOPO"
        mname = {"lasso": "Lasso", "hgb": "HGB"}[m2]
        ax.set_xlabel(f"KRR (G+E), {cvl}"); ax.set_ylabel(f"{mname} (G+E), {cvl}")
        ax.set_title(f"KRR vs {mname}" + (f" | Holm {ptext(t.p_holm)}" if t is not None else ""),
                     fontsize=8.5, pad=3)
        if k == 0:
            ax.legend(fontsize=6.2, loc="upper left", ncol=1, handletextpad=0.15, labelspacing=0.25, borderpad=0.1)

    # C: medians with bootstrap CI
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C", dx=-0.18)
    rng = np.random.default_rng(3)
    labels, meds, los, his, colors = [], [], [], [], []
    for cv in ["random", "lopo"]:
        for m in ["kridge", "lasso", "hgb"]:
            r = p[f"{m}__GE__{cv}"].dropna().values
            resamples = rng.choice(r, size=(20000, len(r)), replace=True)
            bs = np.median(resamples, axis=1)
            labels.append(f"{D_MODEL[m]}\n{D_CV[cv]}"); meds.append(np.median(r))
            los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
            colors.append(C_MODEL[m])
    x = np.arange(len(labels))
    ax.errorbar(x, meds, yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
                fmt="none", ecolor="black", lw=0.7, capsize=1.5)
    ax.scatter(x, meds, s=22, color=colors, zorder=3, edgecolor="white", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Median Pearson r (G+E)")
    margin = max(0.03, 0.05 * (max(his) - min(los)))
    ax.set_ylim(min(los) - margin, max(his) + margin)
    ax.set_title("Model performance (95% bootstrap CI)", fontsize=8.5, pad=3)

    # D: gain from expression by model
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D", dx=-0.18)
    data = []
    for m in ["kridge", "lasso", "hgb"]:
        for cv in ["random", "lopo"]:
            data.append((p[f"{m}__GE__{cv}"] - p[f"{m}__G__{cv}"]).dropna().values)
    pos = [0, 0.6, 1.6, 2.2, 3.2, 3.8]
    bp = ax.boxplot(data, positions=pos, widths=0.42, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=0.8),
                    boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    for i, b in enumerate(bp["boxes"]):
        m = ["kridge", "lasso", "hgb"][i // 2]
        b.set_facecolor(C_MODEL[m]); b.set_alpha(0.35 if i % 2 == 0 else 0.75)
    for i, d in enumerate(data):
        ax.scatter(pos[i] + np.random.default_rng(5).normal(0, 0.03, len(d)), d,
                   s=3, color="#333333", alpha=0.5, lw=0, zorder=3)
    ax.axhline(0, color=GREY, lw=0.5, ls="--")
    ax.set_xticks([0.3, 1.9, 3.5])
    ax.set_xticklabels(["KRR", "Lasso", "HGB"])
    ax.set_ylabel("$\\Delta$r (G+E $-$ G)")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.35, label="Random"),
                       Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.75, label="LOPO")],
              fontsize=7, loc="upper right")
    ax.set_title("Expression gain depends on model and CV", fontsize=8.5, pad=3)

    fig.text(0.01, 0.005, "Points are traits; boxplots summarize paired differences. LOPO, leave-population-out; KRR, kernel ridge regression; HGB, histogram gradient boosting.",
             fontsize=6.5, color=GREY)
    save(fig, "fig4_models")


# ---------------------------------------------------------------- Figure 5
def fig5():
    tr = pd.read_csv(os.path.join(RES, "transfer.csv"))
    ctrl = pd.read_csv(os.path.join(RES, "controls.csv"))
    enr = pd.read_csv(os.path.join(RES, "enrichment.csv"))

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.48, wspace=0.56,
                          left=0.22, right=0.98, top=0.95, bottom=0.12)

    # A: transfer heatmap (kridge)
    ax = fig.add_subplot(gs[0, 0:2])
    panel_label(ax, "A", dx=-0.55)
    k = tr[(tr.model == "kridge")]
    pairs = k[["source", "target", "note"]].drop_duplicates()
    pairs = pairs[pairs.source != "within"]
    M = []
    ylab = []
    for r in pairs.itertuples():
        row = []
        for f in ["G", "E", "GE"]:
            v = k[(k.source == r.source) & (k.target == r.target) & (k.features == f)]
            row.append(v.r.iloc[0] if len(v) else np.nan)
        M.append(row)
        ylab.append(f"{r.source} $\\rightarrow$ {r.target}")
    M = np.array(M)
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.6)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["G", "E", "G+E"])
    ax.set_yticks(np.arange(len(ylab))); ax.set_yticklabels(ylab, fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if M[i, j] > 0.4 or M[i, j] < -0.15 else "black")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cross-study / condition / record prediction (KRR; r)", fontsize=8.5, pad=3)

    # B: within-target reference
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "B", dx=-0.35)
    w = tr[tr.source == "within"]
    w = w[w.features == "GE"]
    ax.barh(np.arange(len(w)), w.r.values, color=C_GE, alpha=0.85, lw=0)
    ax.set_yticks(np.arange(len(w))); ax.set_yticklabels(w.target.values, fontsize=6.3)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Within-target r (KRR G+E)")
    ax.set_title("Same-accession reference", fontsize=8.8, pad=3)

    # C: controls
    ax = fig.add_subplot(gs[1, 0:2])
    panel_label(ax, "C", dx=-0.28)
    ctrl = ctrl.copy()
    traits = list(dict.fromkeys(ctrl.trait))
    offs = {"G": -0.22, "E": 0.0, "GE": 0.22}
    for i, t in enumerate(traits):
        sub = ctrl[ctrl.trait == t]
        for f in ["G", "E", "GE"]:
            s = sub[sub.features == f]
            if not len(s):
                continue
            r = s.real_r.iloc[0]; q = s.null_label_q95.iloc[0]; p = s.p_label.iloc[0]
            x = i + offs[f]
            ax.plot([x, x], [0, q], color="#BDBDBD", lw=2.2, solid_capstyle="butt", zorder=1)
            ax.scatter(x, r, s=16, color=C_FEAT[f], zorder=3, edgecolor="white", lw=0.3)
            if p < 0.05:
                ax.text(x, max(r, q) + 0.03, "*", ha="center", fontsize=9.1)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels([t.replace(" ", "\n", 1) for t in traits], fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("Observed vs label-permutation null (95th percentile)", fontsize=8.8, pad=3)
    ax.set_ylim(top=max(ctrl.real_r.max(), ctrl.null_label_q95.max()) + 0.13)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C_FEAT[f], lw=0, ms=3.5,
                              label=D_FEAT[f]) for f in ["G", "E", "GE"]],
              fontsize=7.0, loc="upper right", ncol=3, handletextpad=0.2, columnspacing=0.7)

    # D: enrichment
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "D", dx=-0.42)
    enr = enr.copy()
    enr["lab"] = enr.features + " " + enr.score
    enr = enr.sort_values(["features", "score", "top_frac"])
    y = np.arange(len(enr))
    ax.scatter(enr.fold_enrich, y, s=8 + 14 * (-np.log10(enr.p.clip(lower=1e-12))),
               color=[C_FEAT[f[0]] if f[0] in "GE" else GREY for f in enr.features],
               alpha=0.85, lw=0)
    ax.set_yticks(y); ax.set_yticklabels([f"{D_FEAT[a]} {b} {int(100*c)}%" for a, b, c in
                                          zip(enr.features, enr.score, enr.top_frac)], fontsize=6.2)
    ax.axvline(1, color=GREY, lw=0.5, ls="--")
    ax.set_xlabel("Fold enrichment")
    ax.set_title("FT16 gene-set enrichment", fontsize=8.8, pad=3)
    ax.invert_yaxis()

    fig.text(0.01, 0.005, "A/B: shared accession panels. C: conditional nulls, fixed tuning. D: dot area increases with $-\\log_{10}P$; nominal P.",
             fontsize=6.5, color=GREY)
    save(fig, "fig5_transfer_controls")


# ---------------------------------------------------------------- Supplement
def figS1(per):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.35, left=0.08, right=0.98, top=0.88, bottom=0.16))
    p = trait_order(per)
    pooled = pd.read_csv(os.path.join(RES, "full", "pooled_metrics.csv"))
    pooled = pooled.drop_duplicates(["trait_id", "model", "features", "cv", "rep"], keep="last")
    pooled = pooled[(pooled.model == "kridge") & (pooled.features == "GE") & (pooled.cv == "random")]
    r2 = pooled.groupby("trait_id").r2.mean().reindex(p.trait_id).to_numpy()
    h = p.h2.to_numpy()
    for k, (accuracy, title, ylabel) in enumerate([
            (p["kridge__GE__random"].to_numpy(), "Correlation accuracy", "Pearson r (KRR G+E, random CV)"),
            (r2, "Predictive $R^2$", "Pooled $R^2$ (KRR G+E, random CV)")]):
        ax = axes[k]
        ax.scatter(h, accuracy, s=16, c=[C_GROUP[g] for g in p.group], edgecolor="white", lw=0.3)
        m = np.isfinite(h) & np.isfinite(accuracy)
        rho, pv = spearmanr(h[m], accuracy[m])
        ax.set_xlabel("REML heritability (h$^2$)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.1)
        ax.text(0.03, 0.97, f"Spearman $\\rho$={rho:.2f}\nDescriptive {ptext(pv)}",
                transform=ax.transAxes, va="top", fontsize=7.2)
        panel_label(ax, "AB"[k], dx=-0.18)
    save(fig, "figS1_heritability_inflation")


def figS2():
    fold = pd.read_csv(os.path.join(RES, "full", "fold_metrics.csv"))
    pop = pd.read_csv(os.path.join(ANA, "per_trait.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.35, left=0.08, right=0.98, top=0.88, bottom=0.17))
    ax = axes[0]
    sub = fold[(fold.cv == "random") & (fold.model == "kridge") & (fold.features.isin(["G", "GE"]))]
    sub = sub[np.isfinite(sub.r2_fold)]
    bins = np.histogram_bin_edges(sub.r2_fold.to_numpy(), bins=40)
    ax.hist(sub[sub.features == "G"].r2_fold, bins=bins, density=True, color=C_G, alpha=0.45, label="G")
    ax.hist(sub[sub.features == "GE"].r2_fold, bins=bins, density=True, color=C_GE, alpha=0.45, label="G+E")
    ax.axvline(0, color=GREY, lw=0.5, ls="--")
    ax.set_xlabel("Fold-level $R^2$"); ax.set_ylabel("Density")
    ax.legend(fontsize=7.8)
    ax.set_title("Fold-level variability (random CV)", fontsize=9.1)
    panel_label(ax, "A", dx=-0.18)

    ax = axes[1]
    rng = np.random.default_rng(1)
    for i, (m, f) in enumerate([("kridge", "G"), ("kridge", "E"), ("kridge", "GE"),
                                ("lasso", "GE"), ("hgb", "GE")]):
        r = pop[f"{m}__{f}__random"].dropna().values
        ax.scatter(np.full(len(r), i) + rng.normal(0, 0.06, len(r)), r, s=5,
                   color=C_FEAT[f], alpha=0.6, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(r)] * 2, color="black", lw=0.9)
    ax.set_xticks(range(5)); ax.set_xticklabels(["KRR G", "KRR E", "KRR G+E", "Lasso G+E", "HGB G+E"], fontsize=7.3)
    ax.set_ylabel("Pearson r (random CV)")
    ax.set_title("Between-trait spread", fontsize=9.1)
    panel_label(ax, "B", dx=-0.18)
    save(fig, "figS2_variability")


def figS3():
    df = pd.read_csv(os.path.join(ANA, "fullpanel_vs_exprpanel.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.35, left=0.09, right=0.98, top=0.88, bottom=0.16))
    for k, (a, b, lab) in enumerate([("fullpanel_random", "kridge__G__random", "Random"),
                                     ("fullpanel_lopo", "kridge__G__lopo", "LOPO")]):
        ax = axes[k]
        ax.scatter(df[a], df[b], s=15, c=[C_GROUP[g] for g in df.group], edgecolor="white", lw=0.3)
        values = df[[a, b]].to_numpy().ravel()
        values = values[np.isfinite(values)]
        lim = ax.get_xlim()
        if len(values):
            margin = max(0.02, 0.05 * np.ptp(values))
            lim = [values.min() - margin, values.max() + margin]
        ax.plot(lim, lim, color=GREY, lw=0.6, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"All genotyped accessions (n up to {int(df.n.max())})")
        ax.set_ylabel("Expression-matched panel")
        rho, pv = spearmanr(df[a], df[b])
        ax.set_title(f"{lab}: $\\rho$={rho:.2f}, {ptext(pv)}", fontsize=9.1)
        panel_label(ax, "AB"[k], dx=-0.18)
    save(fig, "figS3_panel_sensitivity")


def figS4():
    P = Panel()
    meta = pd.read_csv(os.path.join(PROC, "benchmark_traits.csv"))
    fig, axes = plt.subplots(5, 6, figsize=(7.2, 5.8))
    axes = axes.ravel()
    for i, r in enumerate(meta.itertuples()):
        ax = axes[i]
        ids, y, cl = P.trait(r.study_id, r.trait, require_expr=False)
        ax.hist(y, bins=18, color=C_GROUP[r.group], alpha=0.8, lw=0)
        ax.set_title(f"{r.trait_id}\nn={len(y)}, h$^2$={r.h2:.2f}", fontsize=6.8, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.text(0.5, 0.01, "Winsorized (1st-99th percentile) and z-scored trait values",
             ha="center", fontsize=7.2, color=GREY)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.05, hspace=0.55, wspace=0.25)
    save(fig, "figS4_trait_distributions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    per, tests, summ = load_analysis()
    do = args.only
    if do in ("all", "1"):
        fig1(per)
    if do in ("all", "2"):
        fig2(per, tests)
    if do in ("all", "3"):
        fig3(per, tests)
    if do in ("all", "4"):
        fig4(per, tests)
    if do in ("all", "5"):
        fig5()
    if do in ("all", "S1"):
        figS1(per)
    if do in ("all", "S2"):
        figS2()
    if do in ("all", "S3"):
        figS3()
    if do in ("all", "S4"):
        figS4()
