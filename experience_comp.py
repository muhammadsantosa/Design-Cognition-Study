#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm
from statsmodels.stats.multitest import multipletests


metric_cols = [
    "Satisfaction","Quality","Novelty","Mental Demand","Physical Demand",
    "Temporal Demand","Performance","Effort","Frustration","Overall Workload",
    "Stress","Arousal","Valence"
]

def normalize_modality(x):
    x = str(x).strip().lower()
    if x in ("p","phys","physical"): 
        return "P"
    if x in ("d","dig","digital"):   
        return "D"
    raise ValueError(f"Unknown Modality: {x}")

def normalize_constraint(x):
    x = str(x).strip().lower()
    if x in ("o","open"): 
        return "O"
    if x in ("c","con","constrained","constraint"): 
        return "C"
    raise ValueError(f"Unknown Constraint: {x}")

def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 1:
        return np.nan, np.nan, len(a)
    res = wilcoxon(a, b, zero_method="pratt", alternative="two-sided", method="approx")
    p = float(res.pvalue)

    if np.isnan(p) or p <= 0:
        z = 0.0
    else:
        sign = np.sign(np.nanmedian(a - b)) or 1.0
        z = sign * norm.isf(p/2.0)
    r = abs(z) / np.sqrt(len(a))
    return p, r, len(a)

def load_long():
    csv_path = "/Users/rashid/Library/CloudStorage/OneDrive-UniversityofBristol(2)/MSc Data Science/Dissertation/Technical Work/Objective 3/data/design_process_results.csv"
    df = pd.read_csv(csv_path, header=1)
    df["Modality"]   = df["Modality"].map(normalize_modality)
    df["Constraint"] = df["Constraint"].map(normalize_constraint)
    long_df = df.melt(
        id_vars=["participant", "Modality", "Constraint"],
        value_vars=metric_cols,
        var_name="Factor",
        value_name="Value"
    )
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    return long_df

def cell(long_df, M, C, name):
    sub = long_df[(long_df["Modality"]==M) & (long_df["Constraint"]==C)][
        ["participant", "Factor", "Value"]
    ].copy()
    sub = sub.rename(columns={"Value": name})
    return sub

def compute_qvalues_per_comparison(long_df):
    wide = (
        cell(long_df,"P","O","OP")
        .merge(cell(long_df,"D","O","OD"), on=["participant","Factor"], how="outer")
        .merge(cell(long_df,"P","C","CP"), on=["participant","Factor"], how="outer")
        .merge(cell(long_df,"D","C","CD"), on=["participant","Factor"], how="outer")
    )

    wide["P_avg"] = wide[["OP","CP"]].mean(axis=1)
    wide["D_avg"] = wide[["OD","CD"]].mean(axis=1)
    wide["O_avg"] = wide[["OP","OD"]].mean(axis=1)
    wide["C_avg"] = wide[["CP","CD"]].mean(axis=1)

    CONTRASTS = {
        "P vs D": ("P_avg","D_avg"),
        "O vs C": ("O_avg","C_avg"),
        "OP vs OD": ("OP","OD"),
        "CP vs CD": ("CP","CD"),
        "CP vs OP": ("CP","OP"),
        "CD vs OD": ("CD","OD"),
    }

    rows = []
    for factor, g in wide.groupby("Factor", dropna=False):
        for comp, (a_col, b_col) in CONTRASTS.items():
            p, r, n = wilcoxon_paired(g[a_col].values, g[b_col].values)
            rows.append({"Factor":factor, "Comparison":comp, "p":p, "r":r, "N":n})
    out = pd.DataFrame(rows)

    # FDR correction per comparison family (across metrics)
    out["q"] = np.nan
    out["Discovery?"] = "No"
    for comp, sub in out.groupby("Comparison"):
        pvals = sub["p"].fillna(1.0).values
        rej, qvals, _, _ = multipletests(pvals, alpha=0.01, method="fdr_bh")
        out.loc[sub.index, "q"] = qvals
        out.loc[sub.index, "Discovery?"] = np.where(qvals < 0.01, "Yes", "No")
    return out

def bar_group(values, labels, condition_labels, colors, out_path):
    n_metrics, n_series = values.shape
    x = np.arange(n_metrics)
    width = 0.8 / n_series

    plt.figure(figsize=(14,6))
    for i in range(n_series):
        plt.bar(x + (i - (n_series-1)/2)*width, values[:, i],
                width=width, color=colors[i], label=condition_labels[i], alpha=0.9, edgecolor='black')

    plt.xticks(x, labels, rotation=35, ha='right')
    plt.ylabel("Subjective measure scores")
    plt.xlabel("Measured Factor")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_bars_OP_CP_OD_CD(long_df, out_path):
    wide = (
        cell(long_df,"P","O","OP")
        .merge(cell(long_df,"D","O","OD"), on=["participant","Factor"], how="outer")
        .merge(cell(long_df,"P","C","CP"), on=["participant","Factor"], how="outer")
        .merge(cell(long_df,"D","C","CD"), on=["participant","Factor"], how="outer")
    )
    order = metric_cols
    means = np.array([
        [np.nanmean(wide[wide["Factor"]==m][col].values) for col in ["OP","CP","OD","CD"]]
        for m in order
    ])
    condition_labels = ["OP","CP","OD","CD"]
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]
    bar_group(means, order, condition_labels, colors, out_path)

def plot_bars_P_D_O_C(long_df, out_path):
    OP = cell(long_df,"P","O","OP")
    OD = cell(long_df,"D","O","OD")
    CP = cell(long_df,"P","C","CP")
    CD = cell(long_df,"D","C","CD")

    wide = OP.merge(OD,on=["participant","Factor"],how="outer").merge(CP,on=["participant","Factor"],how="outer").merge(CD,on=["participant","Factor"],how="outer")
    wide["P_avg"] = wide[["OP","CP"]].mean(axis=1)
    wide["D_avg"] = wide[["OD","CD"]].mean(axis=1)
    wide["O_avg"] = wide[["OP","OD"]].mean(axis=1)
    wide["C_avg"] = wide[["CP","CD"]].mean(axis=1)

    order = metric_cols
    means = np.array([
        [np.nanmean(wide[wide["Factor"]==m][col].values) 
         for col in ["P_avg","D_avg","O_avg","C_avg"]]
        for m in order
    ])

    condition_labels = ["P","D","O","C"]
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]

    n_metrics = means.shape[0]
    x = np.arange(n_metrics)

    width = 0.18
    offsets = {
        0: -0.5*width, # P
        1: 0.5*width, # D
        2: 1.5*width+0.1, # O 
        3: 2.5*width+0.1 # C
    }

    plt.figure(figsize=(14,6))
    for i in range(4):
        xi = x + offsets[i]
        plt.bar(xi, means[:, i], width=width, 
                color=colors[i], label=condition_labels[i], alpha=0.9, edgecolor="black")

    xtick_positions = x + 0.23  
    plt.xticks(xtick_positions, order, rotation=35, ha='right')
    plt.ylabel("Subjective measure scores")
    plt.xlabel("Measured Factor")
    plt.legend(loc='upper right', ncol=1)  
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# Heatmaps
def heatmap_q(qtable, comparisons, out_path):
    sub = qtable[qtable["Comparison"].isin(comparisons)].copy()
    piv = sub.pivot(index="Comparison", columns="Factor", values="q")
    piv = piv.loc[comparisons, metric_cols]

    plt.figure(figsize=(max(10, 0.7*len(metric_cols)), 1.3*len(comparisons)))
    im = plt.imshow(piv.values, aspect='auto', cmap='Reds', vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, label="q-value")

    plt.yticks(range(len(piv.index)), piv.index)
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=35, ha='right')

    # Significance highlighted in red
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            qv = piv.iloc[i, j]
            txt = "—" if pd.isna(qv) else f"{qv:.3f}"
            color = 'white' if (pd.notna(qv) and qv > 0.35) else 'black'
            plt.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)
            if pd.notna(qv) and qv < 0.01:
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                                  fill=False, edgecolor='green', linewidth=3))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    out_dir = "/Users/rashid/Library/CloudStorage/OneDrive-UniversityofBristol(2)/MSc Data Science/Dissertation/Technical Work/Objective 3/results/designer_experience"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    long_df = load_long()

    # Bar charts 
    fig1 = f"{out_dir}/interaction_barchart.png"
    fig2 = f"{out_dir}/main_barchart.png"
    plot_bars_OP_CP_OD_CD(long_df, fig1)
    plot_bars_P_D_O_C(long_df, fig2)
    print("Saved bar charts:")
    print(" -", fig1)
    print(" -", fig2)

    # Stats & heatmaps
    qtable = compute_qvalues_per_comparison(long_df)
    q_csv = f"{out_dir}/subjective_measurement_statistics.csv"
    qtable.to_csv(q_csv, index=False)
    print("Saved q-values:", q_csv)

    main_comps = ["P vs D","O vs C"]
    inter_comps = ["OP vs OD","CP vs CD","CP vs OP","CD vs OD"]

    hm1 = f"{out_dir}/heatmap_main_effects.png"
    hm2 = f"{out_dir}/heatmap_interactions.png"
    heatmap_q(qtable, main_comps, hm1)
    heatmap_q(qtable, inter_comps, hm2)

    print("Saved heatmaps:")
    print(" -", hm1)
    print(" -", hm2)

if __name__ == "__main__":
    main()