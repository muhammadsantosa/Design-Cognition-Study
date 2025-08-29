#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from pathlib import Path
import matplotlib.pyplot as plt

np.random.seed(42)


def load_data(file_path: Path) -> pd.DataFrame:
    print(f"Attempting to load data from {file_path}")
    df = pd.read_csv(file_path) #Loading subjectstats.csv
    df = df[df['type'].str.lower() == 'hbo'] #Filtering for HbO channels only
    print(f"Successfully loaded {file_path}")
    return df

def parse_condition(cond_str: str):
    #Getting condition name and decile from the condition string (e.g, phys_const_1)
    try:
        parts = cond_str.split('_')
        name_part = parts[0] + '_' + parts[1]
        decile = int(parts[2])
        return name_part, decile
    except Exception:
        return None, None


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")
    df[['condition_name', 'decile']] = df['cond'].apply(lambda x: pd.Series(parse_condition(x)))
    df = df.dropna(subset=['condition_name', 'decile']).copy()
    df['decile'] = df['decile'].astype(int)

    # Obtaining modality and constraint based on abbreviations
    df['modality'] = df['condition_name'].apply(lambda x: 'Physical' if 'phys' in x else 'Digital')
    df['constraint'] = df['condition_name'].apply(lambda x: 'Constrained' if 'const' in x else 'Open')

    # Condition labels
    df['conditions'] = df['constraint'].str[0] + df['modality'].str[0] 

    # Channel key (for ROI grouping later) (source-detector)
    df['channel'] = df['source'].astype(str) + '-' + df['detector'].astype(str)

    print("Preprocessing completed.")
    return df

def calculate_roi_activation(df: pd.DataFrame, rois: dict) -> pd.DataFrame:
    #Aggregating channels beta activation values based on ROI
    print("Calculating ROI activation...")
    if df.empty:
        print("Empty dataframe passed to calculate_roi_activation.")
        return pd.DataFrame()

    roi_chunks = []
    for roi_name, channels in rois.items():
        print(f"Processing ROI: {roi_name}")
        df_roi = df[df['channel'].isin(channels)].copy()
        if df_roi.empty:
            continue
        # Average over channels within ROI
        g = (df_roi.groupby(['subject', 'modality', 'constraint', 'decile', 'conditions'], as_index=False)['beta'].mean())
        g['roi'] = roi_name
        roi_chunks.append(g)

    if not roi_chunks:
        print("No ROI data found with the provided channel lists.")
        return pd.DataFrame()

    out = pd.concat(roi_chunks, ignore_index=True)
    print("ROI activation calculation completed.")
    return out


def run_overall_rm_anova(df_roi: pd.DataFrame, out_dir: Path, roi_name: str):
    df_avg = (df_roi.groupby(['subject', 'modality', 'constraint'], as_index=False)['beta'].mean())
    aov = pg.rm_anova(data=df_avg, dv='beta', within=['modality', 'constraint'], subject='subject', detailed=True)
    aov_out = out_dir / f"anova_overall_{roi_name}.csv"
    aov.to_csv(aov_out, index=False)
    print(f"Saved: {aov_out}")

    posthocs_overall_1 = pg.pairwise_tests(data=df_avg, dv='beta', within=['modality', 'constraint'], subject='subject')
    posthocs_overall_1.to_csv(out_dir / f"posthocs_overall_{roi_name}_1.csv", index=False)

    posthocs_overall_2 = pg.pairwise_tests(data=df_avg, dv='beta', within=['constraint', 'modality'], subject='subject')
    posthocs_overall_2.to_csv(out_dir / f"posthocs_overall_{roi_name}_2.csv", index=False)

    print(f"Saved: posthocs_overall_{roi_name}.csv")  

    cond_means = (df_avg.assign(conditions=df_avg['constraint'].str[0] + df_avg['modality'].str[0]).groupby('conditions', as_index=False)['beta'].mean().rename(columns={'beta': 'mean_beta'}))
    cond_means.to_csv(out_dir / f"means_overall_{roi_name}_by_condition.csv", index=False)

    constraint_means = (df_avg.groupby('constraint', as_index=False)['beta'].mean().rename(columns={'beta': 'mean_beta'}))
    constraint_means.to_csv(out_dir / f"means_overall_{roi_name}_by_constraint.csv", index=False)

    modality_means   = (df_avg.groupby('modality', as_index=False)['beta'].mean().rename(columns={'beta': 'mean_beta'}))
    modality_means.to_csv(out_dir / f"means_overall_{roi_name}_by_modality.csv", index=False)

    condition_pairs = [('OP','OD'),('OP','CP'),('CP','CD'),('OD','CD')]
    m = cond_means.set_index('conditions')['mean_beta']
    rows = []
    for A,B in condition_pairs:
        if A in m and B in m:
            rows.append({'comparison': f'{A}_vs_{B}', 
                         'mean_A': float(m[A]), 
                         'mean_B': float(m[B]),
                         'mean_diff': float(m[A]-m[B])})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"means_overall_{roi_name}_pairwise.csv", index=False)

def per_decile_interaction_effects(df_roi: pd.DataFrame, out_dir: Path, roi_name: str):
    roi_dir = out_dir / f"per_decile_ttests_{roi_name}"
    roi_dir.mkdir(parents=True, exist_ok=True)

    deciles = sorted(df_roi['decile'].unique())

    agg = (df_roi.groupby(['conditions', 'decile'], as_index=False).agg(mean_beta=('beta', 'mean'),sd_beta=('beta', 'std'),n=('beta', 'count')))
    agg['sem_beta'] = agg['sd_beta'] / np.sqrt(agg['n'].replace(0, np.nan))

    def get_paired(df_slice, cond_a, cond_b):
        da = df_slice[df_slice['conditions'] == cond_a][['subject', 'beta']].rename(columns={'beta': 'beta_a'})
        db = df_slice[df_slice['conditions'] == cond_b][['subject', 'beta']].rename(columns={'beta': 'beta_b'})
        merged = pd.merge(da, db, on='subject', how='inner')  
        return merged
    
    comparisons = [
        ('OP_vs_OD', 'OP', 'OD'),
        ('OP_vs_CP', 'OP', 'CP'),
        ('CP_vs_CD', 'CP', 'CD'),
        ('OD_vs_CD', 'OD', 'CD'),
    ]
    for comp_name, cond_a, cond_b in comparisons:
        rows = []
        pvals = []

        for d in deciles:
            df_d = df_roi[df_roi['decile'] == d]
            paired = get_paired(df_d, cond_a, cond_b)

            n_pairs = len(paired)
            if n_pairs < 2:
                rows.append({
                    'roi': roi_name, 'comparison': comp_name, 'decile': d,
                    'n_pairs': n_pairs, 'mean_A': np.nan, 'mean_B': np.nan,
                    'mean_diff': np.nan, 't': np.nan, 'dof': np.nan, 'p': np.nan,
                    'q': np.nan, 'ci95%_low': np.nan, 'ci95%_high': np.nan
                })
                pvals.append(np.nan)
                continue

            res = pg.ttest(paired['beta_a'], paired['beta_b'], paired=True)
            res_row = res.iloc[0]

            mean_a = paired['beta_a'].mean()
            mean_b = paired['beta_b'].mean()
            mean_diff = mean_a - mean_b

            ci_low, ci_high = np.nan, np.nan
            ci_val = res_row.get('CI95%')
            if isinstance(ci_val, (tuple, list, np.ndarray)) and len(ci_val) == 2:
                ci_low, ci_high = float(ci_val[0]), float(ci_val[1])
            elif isinstance(ci_val, str):
                try:
                    s = ci_val.strip().strip('[]')
                    parts = [float(x) for x in s.split(',')]
                    if len(parts) == 2:
                        ci_low, ci_high = parts
                except Exception:
                    pass

            rows.append({
                'roi': roi_name, 'comparison': comp_name, 'decile': d,
                'n_pairs': n_pairs,
                'mean_A': mean_a, 'mean_B': mean_b, 'mean_diff': mean_diff,
                't': res_row['T'], 'dof': res_row['dof'],
                'p': res_row['p-val'],
                'q': np.nan, 
                'ci95%_low': ci_low, 'ci95%_high': ci_high,
                'cohen_dz': res_row['cohen-d'] if 'cohen-d' in res.columns else np.nan
            })
            pvals.append(res_row['p-val'])

        res_df = pd.DataFrame(rows)
        out_csv = roi_dir / f"{roi_name}_{comp_name}_per_decile_ttests.csv"
        res_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        #Plotting comparisons   
        df_plot = df_roi[df_roi['conditions'].isin([cond_a, cond_b])].copy()
        df_plot['condition'] = df_plot['constraint'].str[0] + df_plot['modality'].str[0]

        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")

        condition_order = ["OD", "OP", "CP", "CD"]
        condition_palette = {
            "OD": "#1f77b4", 
            "OP": "#ff7f0e", 
            "CP": "#2ca02c",
            "CD": "#d62728",  
        }
        condition_markers = {
            "OD": "X",  
            "OP": "o",
            "CP": "s",
            "CD": "^",
        }
        ax = sns.lineplot(
            data=df_plot,
            x='decile',
            y='beta',
            hue='condition',
            style='condition',
            hue_order=condition_order,
            style_order=condition_order,
            palette=condition_palette,
            markers=condition_markers,
            dashes=False,
            errorbar='se'
        )

        if not res_df.empty and 'p' in res_df.columns:
            sigs = res_df[(res_df['p'] < 0.05) & (res_df['p'].notna())]
            if not sigs.empty:
                agg_plot = agg[agg['conditions'].isin([cond_a, cond_b])].copy()
                for _, r in sigs.iterrows():
                    dec = int(r['decile'])
                    for c in [cond_a, cond_b]:
                        y_vals = agg_plot[(agg_plot['conditions'] == c) & (agg_plot['decile'] == dec)]['mean_beta'].values
                        if len(y_vals) == 1 and pd.notnull(y_vals[0]) and np.isfinite(y_vals[0]):
                            ax.scatter(dec, y_vals[0], facecolors='none', edgecolors='black', s=180, linewidths=2, zorder=6)

        ax.set_xlabel("Time (Decile)", fontsize=12)
        ax.set_ylabel("Activation Level (Beta Value)", fontsize=12)
        ax.set_xticks(list(range(1, 11)))
        ax.axhline(0, ls='--', color='black', linewidth=0.8)
        ax.legend(title='Condition', fontsize=14)
        plt.tight_layout()

        out_png = roi_dir / f"{roi_name}_{comp_name}_per_decile_plot.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved: {out_png}")

def per_decile_main_effects(df_roi: pd.DataFrame, out_dir: Path, roi_name: str):
    eff_dir = out_dir / f"per_decile_main_effects_{roi_name}"
    eff_dir.mkdir(parents=True, exist_ok=True)

    deciles = sorted(df_roi['decile'].unique())

    #Constraint main effects
    rows_con, pvals_con = [], []
    for d in deciles:
        dsub = df_roi[df_roi['decile'] == d]
        oc = (dsub.groupby(['subject', 'constraint'])['beta'].mean().unstack('constraint'))
        if {'Open', 'Constrained'}.issubset(set(oc.columns)):
            oc = oc[['Open', 'Constrained']].dropna()
            n_pairs = len(oc)
            if n_pairs >= 2:
                res = pg.ttest(oc['Open'], oc['Constrained'], paired=True).iloc[0]
                rows_con.append({
                    'roi': roi_name, 'effect': 'constraint', 'decile': d, 'n_pairs': n_pairs,
                    'mean_Open': oc['Open'].mean(), 'mean_Constrained': oc['Constrained'].mean(),
                    'mean_diff(Open-Constrained)': oc['Open'].mean() - oc['Constrained'].mean(),
                    't': res['T'], 'dof': res['dof'], 'p': res['p-val'], 'q': np.nan,
                    'cohen_dz': res.get('cohen-d', np.nan)
                })
                pvals_con.append(res['p-val'])
            else:
                rows_con.append({
                    'roi': roi_name, 'effect': 'constraint', 'decile': d, 'n_pairs': n_pairs,
                    'mean_Open': np.nan, 'mean_Constrained': np.nan,
                    'mean_diff(Open-Constrained)': np.nan,
                    't': np.nan, 'dof': np.nan, 'p': np.nan, 'q': np.nan, 'cohen_dz': np.nan
                })
                pvals_con.append(np.nan)

    df_con = pd.DataFrame(rows_con)
    con_csv = eff_dir / f"{roi_name}_main_effect_constraint_per_decile_ttests.csv"
    df_con.to_csv(con_csv, index=False)
    print(f"Saved: {con_csv}")

    # Plottting constraint comparisons
    plot_con = (df_roi.groupby(['subject', 'decile', 'constraint'])['beta']
                      .mean()
                      .reset_index())
    if not plot_con.empty:
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(data=plot_con, x='decile', y='beta',
                          hue='constraint', style='constraint',
                          markers=True, dashes=False, errorbar='se')

        if not df_con.empty and 'p' in df_con.columns:
            sigs = df_con[(df_con['p'] < 0.05) & df_con['p'].notna()]
            if not sigs.empty:
                agg = (plot_con.groupby(['constraint', 'decile'], as_index=False)['beta']
                              .mean().rename(columns={'beta': 'mean_val'}))
                for _, r in sigs.iterrows():
                    dec = int(r['decile'])
                    for lvl in ['Open', 'Constrained']:
                        y = agg[(agg['constraint'] == lvl) & (agg['decile'] == dec)]['mean_val']
                        if len(y) == 1 and pd.notnull(y.iloc[0]) and np.isfinite(y.iloc[0]):
                            ax.scatter(dec, y.iloc[0], facecolors='none', edgecolors='black',
                                       s=180, linewidths=2, zorder=6)

        ax.set_xlabel("Time (Decile)"); ax.set_ylabel("Activation (Beta)")
        ax.set_xticks(list(range(1, 11))); ax.legend(title='Constraint', fontsize=14)
        ax.axhline(0, ls='--', color='black', linewidth=0.8)
        plt.tight_layout()
        out_png = eff_dir / f"{roi_name}_main_effect_constraint_per_decile_plot.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved: {out_png}")

    # Modality main effects
    rows_mod, pvals_mod = [], []
    for d in deciles:
        dsub = df_roi[df_roi['decile'] == d]
        pdm = (dsub.groupby(['subject', 'modality'])['beta'].mean().unstack('modality'))
        if {'Physical', 'Digital'}.issubset(set(pdm.columns)):
            pdm = pdm[['Physical', 'Digital']].dropna()
            n_pairs = len(pdm)
            if n_pairs >= 2:
                res = pg.ttest(pdm['Physical'], pdm['Digital'], paired=True).iloc[0]
                rows_mod.append({
                    'roi': roi_name, 'effect': 'modality', 'decile': d, 'n_pairs': n_pairs,
                    'mean_Physical': pdm['Physical'].mean(), 'mean_Digital': pdm['Digital'].mean(),
                    'mean_diff(Physical-Digital)': pdm['Physical'].mean() - pdm['Digital'].mean(),
                    't': res['T'], 'dof': res['dof'], 'p': res['p-val'], 'q': np.nan,
                    'cohen_dz': res.get('cohen-d', np.nan)
                })
                pvals_mod.append(res['p-val'])
            else:
                rows_mod.append({
                    'roi': roi_name, 'effect': 'modality', 'decile': d, 'n_pairs': n_pairs,
                    'mean_Physical': np.nan, 'mean_Digital': np.nan,
                    'mean_diff(Physical-Digital)': np.nan,
                    't': np.nan, 'dof': np.nan, 'p': np.nan, 'q': np.nan, 'cohen_dz': np.nan
                })
                pvals_mod.append(np.nan)

    df_mod = pd.DataFrame(rows_mod)
    mod_csv = eff_dir / f"{roi_name}_main_effect_modality_per_decile_ttests.csv"
    df_mod.to_csv(mod_csv, index=False)
    print(f"Saved: {mod_csv}")

    # Plotting modality comparisons
    plot_mod = (df_roi.groupby(['subject', 'decile', 'modality'])['beta'].mean().reset_index())
    if not plot_mod.empty:
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(data=plot_mod, x='decile', y='beta',
                          hue='modality', style='modality',
                          markers=True, dashes=False, errorbar='se')

        if not df_mod.empty and 'p' in df_mod.columns:
            sigs = df_mod[(df_mod['p'] < 0.05) & df_mod['p'].notna()]
            if not sigs.empty:
                agg = (plot_mod.groupby(['modality', 'decile'], as_index=False)['beta']
                              .mean().rename(columns={'beta': 'mean_val'}))
                for _, r in sigs.iterrows():
                    dec = int(r['decile'])
                    for lvl in ['Physical', 'Digital']:
                        y = agg[(agg['modality'] == lvl) & (agg['decile'] == dec)]['mean_val']
                        if len(y) == 1 and pd.notnull(y.iloc[0]) and np.isfinite(y.iloc[0]):
                            ax.scatter(dec, y.iloc[0], facecolors='none', edgecolors='black',
                                       s=180, linewidths=2, zorder=6)

        ax.set_xlabel("Time (Decile)"); ax.set_ylabel("Activation (Beta)")
        ax.set_xticks(list(range(1, 11))); ax.legend(title='Modality', fontsize=14)
        ax.axhline(0, ls='--', color='black', linewidth=0.8)
        plt.tight_layout()
        out_png = eff_dir / f"{roi_name}_main_effect_modality_per_decile_plot.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved: {out_png}")


def run_activation_analysis(base_dir: Path, out_dir: Path):
    print("Starting activation analysis (per-decile t-tests)...")

    subject_stats_path = base_dir / "SubjectStats.csv"
    df = load_data(subject_stats_path)
    if df.empty:
        print("No data loaded. Exiting.")
        return

    df = preprocessing(df)

    rois = {
        "FPC": ['2-3', '5-3', '5-6', '8-6'],
        "OFC": ['2-3', '5-3', '5-6', '8-6', '3-3', '5-4', '6-6', '3-4', '6-4'],
        "DLPFC": ['1-1', '2-1', '3-1', '2-2', '3-4', '4-2', '4-4', '4-5', '6-4', '6-7', '7-5', '7-7', '8-7'],
        "Right_DLPFC": ['4-5', '6-4', '6-7', '7-5', '7-7', '8-7'],
        "Left_DLPFC": ['1-1', '2-1', '3-1', '2-2', '3-4', '4-2']
    }

    df_roi = calculate_roi_activation(df, rois)
    if df_roi.empty:
        print("No ROI activation data available for analysis.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Loop ROIs
    for roi_name in rois.keys():
        print(f"\n{'='*30}\n   ANALYSIS FOR ROI: {roi_name}\n{'='*30}")
        df_this = df_roi[df_roi['roi'] == roi_name].copy()
        if df_this.empty:
            print(f"No data for ROI {roi_name}. Skipping.")
            continue

        # Overall analysis
        run_overall_rm_anova(df_this, out_dir, roi_name)

        # Interaction Ef
        per_decile_interaction_effects(df_this, out_dir, roi_name)

        # Per-decile main effects 
        per_decile_main_effects(df_this, out_dir, roi_name)


def main():
    parser = argparse.ArgumentParser(description="Per-decile paired t-tests for fNIRS activation analysis")
    parser.add_argument(
        "--base-dir", "-b",
        type=Path,
        default=Path("/Users/rashid/Library/CloudStorage/OneDrive-UniversityofBristol(2)/MSc Data Science/Dissertation/Technical Work/Objective 2/data/DESIGN2024_Analysis"),
        help="Root folder containing SubjectStats.csv"
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=Path,
        default=Path("/Users/rashid/Library/CloudStorage/OneDrive-UniversityofBristol(2)/MSc Data Science/Dissertation/Technical Work/Objective 2/data/results/activation_analysis_ttests"),
        help="Output directory for CSVs and figures"
    )
    args = parser.parse_args()
    run_activation_analysis(args.base_dir, args.out_dir)

if __name__ == "__main__":
    main()
