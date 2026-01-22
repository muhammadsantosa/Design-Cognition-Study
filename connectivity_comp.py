#!/usr/bin/env python3

import os
import mne
import json
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import pingouin as pg
from pathlib import Path
import matplotlib.pyplot as plt
from mne.preprocessing.nirs import scalp_coupling_index as sci
from mne_nirs.signal_enhancement import short_channel_regression
from mne_nirs.channels import get_short_channels, get_long_channels
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.preprocessing.nirs import temporal_derivative_distribution_repair as tddr

np.random.seed(42)

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

def load_data(data_dir: Path):
    description_files = glob.glob(os.path.join(data_dir, "**/*description.json"), recursive=True)
    subjects_to_exclude = ['Henrikke_Delete', '550', '56']

    subject_files = {}
    for desc_file in description_files:
        try:
            with open(desc_file, 'r') as f:
                subject_id = json.load(f)['subject']
                snirf_path = desc_file.replace('_description.json', '.snirf')
                if os.path.exists(snirf_path):
                    subject_files.setdefault(subject_id, {'snirf': [], 'desc': []})
                    subject_files[subject_id]['snirf'].append(snirf_path)
                    subject_files[subject_id]['desc'].append(desc_file)
        except Exception as e:
            print(f"Could not process description file {desc_file}: {e}")

    raw_data_list = []
    for subject_id, files in subject_files.items():
        if subject_id not in subjects_to_exclude:
            try:
                snirf_file_path = files['snirf'][0]
                raw = mne.io.read_raw_snirf(snirf_file_path, preload=True)
                raw.info['subject_info'] = {'his_id': subject_id}
                raw_data_list.append(raw)
            except Exception as e:
                print(f"Could not load {snirf_file_path} for subject {subject_id}: {e}")
    return raw_data_list


def preprocessing(snirf_list, sci_threshold=0.8, l_freq=0.01, h_freq=0.2, ppf=0.1):
    master_bad_channels = set()

    # Identify bad channels by SCI on TDDR-corrected OD
    for i, raw in enumerate(snirf_list):
        print(f"Scanning for bad channels in participant {raw.info['subject_info']['his_id']} ({i+1}/{len(snirf_list)})...")
        try:
            raw_copy = raw.copy()
            raw_od = optical_density(raw_copy)
            raw_od_corrected = tddr(raw_od)
            long_channel_names = get_long_channels(raw_od_corrected).ch_names
            sci_values = sci(raw_od_corrected)
            bad_sci_names = []
            for idx, ch_name in enumerate(raw_od_corrected.ch_names):
                if (sci_values[idx] < sci_threshold) and (ch_name in long_channel_names):
                    bad_sci_names.append(ch_name)
            master_bad_channels.update(bad_sci_names)
            print(f"  - Found {len(bad_sci_names)} bad long channels. Master list size: {len(master_bad_channels)}")
        except Exception as e:
            print(f"!!! FAILED for participant {raw.info['subject_info']['his_id']}: {e}")

    def _crop_task_window(raw_in, pre=15.0, post=15.0):
        if len(raw_in.annotations) == 0:
            return raw_in
        sel = [i for i, d in enumerate(raw_in.annotations.description) if str(d).strip() in ['OP', 'CP', 'OD', 'CD']]
        if sel:
            on = raw_in.annotations.onset[sel]
            du = raw_in.annotations.duration[sel]
            tmin = float(np.min(on))
            tmax = float(np.max(on + du))
        else:
            tmin = float(np.min(raw_in.annotations.onset))
            tmax = float(np.max(raw_in.annotations.onset + raw_in.annotations.duration))

        tmin = tmin - pre
        tmax = tmax + post

        data_start = 0.0
        data_end = float(raw_in.times[-1])
        tmin = max(data_start, tmin)
        tmax = min(data_end, tmax)

        if tmax <= tmin:
            tmin = max(data_start, data_end - 60.0) 
            tmax = data_end

        return raw_in.copy().crop(tmin=tmin, tmax=tmax)

    
    processed_raw_list = []
    for i, raw in enumerate(snirf_list):
        print(f"Processing participant {raw.info['subject_info']['his_id']} ({i+1}/{len(snirf_list)})...")
        try:
            raw_copy = raw.copy()
            rename_mapping = {'1': 'OP', '1.0': 'OP', 
                              '2': 'CP', '2.0': 'CP', 
                              '3': 'OD', '3.0': 'OD', 
                              '4': 'CD', '4.0': 'CD'}
            new_descriptions = [rename_mapping.get(str(d).strip(), str(d).strip()) for d in raw_copy.annotations.description]
            raw_copy.annotations.description = new_descriptions
            raw_copy = _crop_task_window(raw_copy, pre=15.0, post=15.0)
            raw_copy.drop_channels(list(master_bad_channels), on_missing='ignore')
            print(f"  - Dropped bad quality channels. Remaining channels: {len(raw_copy.ch_names)}")

            raw_od = optical_density(raw_copy)
            raw_od_corrected = tddr(raw_od)
            od_corrected_regressed = short_channel_regression(raw_od_corrected)
            short_ch_info = get_short_channels(od_corrected_regressed)
            od_corrected_regressed.drop_channels(short_ch_info.ch_names, on_missing='ignore')
            print(f"  - Explicitly dropped short channels. Remaining channels: {len(od_corrected_regressed.ch_names)}")

            raw_hemo = beer_lambert_law(od_corrected_regressed, ppf=ppf)
            raw_hemo.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
            processed_raw_list.append(raw_hemo)
            print(f"Finished processing participant {raw.info['subject_info']['his_id']}.")
        except Exception as e:
            print(f"!!! FAILED for participant {raw.info['subject_info']['his_id']}: {e}")

    return processed_raw_list


def create_deciles(hemo_list):
    decile_raw_list = []
    for raw_hemo in hemo_list:
        sfreq = float(raw_hemo.info['sfreq'])
        eps = 1.0 / sfreq
        raw_end = float(raw_hemo.times[-1])

        new_onset, new_dur, new_desc = [], [], []
        orig_time = raw_hemo.annotations.orig_time

        for i in range(len(raw_hemo.annotations)):
            annot = raw_hemo.annotations[i]
            cond = str(annot['description'])
            if cond not in ['OP', 'CP', 'OD', 'CD']:
                continue

            start = float(annot['onset'])
            total = float(annot['duration'])
            total = max(0.0, min(total, raw_end - start))

            if total < 5 * eps:
                continue

            dec_dur = total / 10.0
            for k in range(10):
                s = start + k * dec_dur
                e = min(s + dec_dur, raw_end - 0.5 * eps) 
                if e - s < 2 * eps:
                    continue
                new_onset.append(s)
                new_dur.append(e - s)
                new_desc.append(f"{cond}/{k+1}")

        decile_annotations = mne.Annotations(
            onset=new_onset, duration=new_dur, description=new_desc, orig_time=orig_time
        )

        raw_copy = raw_hemo.copy()
        raw_copy.set_annotations(decile_annotations)
        decile_raw_list.append(raw_copy)

    return decile_raw_list


def calc_corr_matrix(decile_raw_list):
    all_connectivity_results = []
    hbo_decile_raw_list = [raw.copy().pick('hbo') for raw in decile_raw_list]

    for raw_hbo in hbo_decile_raw_list:
        subject_id = raw_hbo.info['subject_info']['his_id']
        print(f"Calculating matrices for subject: {subject_id}")
        ann = raw_hbo.annotations
        if len(ann) == 0:
            print(f"  - No annotations found for subject {subject_id}, skipping.")
            continue

        sfreq = float(raw_hbo.info['sfreq'])
        eps = 1.0 / sfreq
        raw_end = float(raw_hbo.times[-1])

        for i in range(len(ann)):
            desc = str(ann[i]['description'])
            if '/' not in desc:
                continue
            cond, dec = desc.split('/')
            try:
                t0 = float(ann[i]['onset'])
                dur = float(ann[i]['duration'])

                t0 = max(0.0, min(t0, raw_end - eps))
                t1 = min(t0 + max(0.0, dur), raw_end - 0.5 * eps)
                if t1 <= t0 + 2 * eps:
                    continue

                seg = raw_hbo.copy().crop(tmin=t0, tmax=t1)
                X = seg.get_data()  

                if X.shape[1] < 5:
                    continue

                corr = np.corrcoef(X)
                all_connectivity_results.append({
                    'subject_id': subject_id,
                    'condition': cond,
                    'decile': int(dec),
                    'correlation_matrix': corr,
                    'channel_names': seg.ch_names
                })
                print(f"Correlation Matrix Size: {corr.shape}")
            except Exception as e:
                print(f"  - Could not process decile '{desc}' for subject {subject_id}: {e}")

    return all_connectivity_results



def network_analysis(out_dir: Path, all_connectivity_results):
    all_network_metrics = []
    absolute_threshold = 0.8

    for result in all_connectivity_results:
        matrix = result['correlation_matrix']
        abs_matrix = np.abs(matrix)
        np.fill_diagonal(abs_matrix, 0)
        binary_matrix = (abs_matrix > absolute_threshold).astype(int)
        G = nx.from_numpy_array(binary_matrix)
        density = nx.density(G)
        avg_degree_centrality = np.mean(list(nx.degree_centrality(G).values()))
        avg_clustering = nx.average_clustering(G)
        global_efficiency = nx.global_efficiency(G)

        metrics = {
            'subject_id': result['subject_id'],
            'condition': result['condition'],
            'decile': result['decile'],
            'density': density,
            'avg_degree_centrality': avg_degree_centrality,
            'avg_clustering': avg_clustering,
            'global_efficiency': global_efficiency,
        }
        all_network_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_network_metrics)
    out_path = out_dir / 'network_metrics.csv'
    df_metrics.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df_metrics


def _ensure_condition_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['modality'] = df['condition'].apply(lambda s: 'Physical' if s.endswith('P') else 'Digital')
    df['constraint'] = df['condition'].apply(lambda s: 'Open' if s.startswith('O') else 'Constrained')
    df['conditions'] = df['condition']  # OP/CP/OD/CD
    return df


def _paired_decile_ttests_for_metric(df_m: pd.DataFrame, metric: str, out_dir: Path):
    metric_dir = out_dir / f"per_decile_ttests_{metric}"
    metric_dir.mkdir(parents=True, exist_ok=True)

    deciles = sorted(df_m['decile'].unique())

    agg = (df_m.groupby(['conditions', 'decile'], as_index=False)
              .agg(mean_val=(metric, 'mean'), sd_val=(metric, 'std'), n=('subject_id', 'count')))
    agg['se_val'] = agg['sd_val'] / np.sqrt(agg['n'].replace(0, np.nan))

    def get_paired(df_slice, cond_a, cond_b):
        da = df_slice[df_slice['conditions'] == cond_a][['subject_id', metric]].rename(columns={metric: 'a'})
        db = df_slice[df_slice['conditions'] == cond_b][['subject_id', metric]].rename(columns={metric: 'b'})
        return pd.merge(da, db, on='subject_id', how='inner')


    comparisons = [
        ("OP_vs_OD", "OP", "OD"),
        ("OP_vs_CP", "OP", "CP"),
        ("CP_vs_CD", "CP", "CD"),
        ("OD_vs_CD", "OD", "CD"),
    ]

    for comp_name, cond_a, cond_b in comparisons:
        rows, pvals = [], []
        for d in deciles:
            df_d = df_m[df_m['decile'] == d]
            paired = get_paired(df_d, cond_a, cond_b)
            n_pairs = len(paired)
            if n_pairs < 2:
                rows.append({'metric': metric, 'comparison': comp_name, 'decile': d, 'n_pairs': n_pairs,
                             'mean_A': np.nan, 'mean_B': np.nan, 'mean_diff': np.nan,
                             't': np.nan, 'dof': np.nan, 'p': np.nan, 'q': np.nan,
                             'ci95%_low': np.nan, 'ci95%_high': np.nan, 'cohen_dz': np.nan})
                pvals.append(np.nan)
                continue
            res = pg.ttest(paired['a'], paired['b'], paired=True)
            res_row = res.iloc[0]
            mean_a, mean_b = paired['a'].mean(), paired['b'].mean()
            mean_diff = mean_a - mean_b
            ci_low, ci_high = np.nan, np.nan
            ci_val = res_row.get('CI95%')
            if isinstance(ci_val, (tuple, list, np.ndarray)) and len(ci_val) == 2:
                ci_low, ci_high = float(ci_val[0]), float(ci_val[1])
            elif isinstance(ci_val, str):
                try:
                    s = ci_val.strip().strip('[]'); parts = [float(x) for x in s.split(',')]
                    if len(parts) == 2: ci_low, ci_high = parts
                except Exception:
                    pass
            rows.append({'metric': metric, 'comparison': comp_name, 'decile': d, 'n_pairs': n_pairs,
                         'mean_A': mean_a, 'mean_B': mean_b, 'mean_diff': mean_diff,
                         't': res_row['T'], 'dof': res_row['dof'], 'p': res_row['p-val'], 'q': np.nan,
                         'ci95%_low': ci_low, 'ci95%_high': ci_high,
                         'cohen_dz': res_row['cohen-d'] if 'cohen-d' in res.columns else np.nan})
            pvals.append(res_row['p-val'])

        res_df = pd.DataFrame(rows)
        out_csv = metric_dir / f"{metric}_{comp_name}_per_decile_ttests.csv"
        res_df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        # Plot
        df_plot = df_m[df_m['conditions'].isin([cond_a, cond_b])].copy()
        df_plot['condition'] = df_plot['conditions']
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(data=df_plot, x='decile', y=metric,
                          hue='condition', style='condition',
                          hue_order=condition_order, style_order=condition_order,
                          palette=condition_palette, markers=condition_markers,
                          dashes=False, errorbar='se')

        # Mark significant deciles with black rings on the mean points
        if not res_df.empty and 'p' in res_df.columns:
            sigs = res_df[(res_df['p'] < 0.05) & (res_df['p'].notna())]
            if not sigs.empty:
                agg_plot = agg[agg['conditions'].isin([cond_a, cond_b])].copy()
                for _, r in sigs.iterrows():
                    dec = int(r['decile'])
                    for c in [cond_a, cond_b]:
                        y_vals = agg_plot[(agg_plot['conditions'] == c) & (agg_plot['decile'] == dec)]['mean_val'].values
                        if len(y_vals) == 1 and pd.notnull(y_vals[0]) and np.isfinite(y_vals[0]):
                            ax.scatter(dec, y_vals[0], facecolors='none', edgecolors='black', s=180, linewidths=2, zorder=6)

        ax.set_title(f"Temporal Dynamics of {metric.replace('_',' ').title()} ({cond_a} vs {cond_b})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (Decile)", fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xticks(list(range(1, 11)))
        ax.legend(title='Condition')
        plt.tight_layout()

        out_png = metric_dir / f"{metric}_{comp_name}_per_decile_plot.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved: {out_png}")

def _per_decile_main_effects(df_m: pd.DataFrame, metric: str, out_dir: Path):

    eff_dir = out_dir / f"per_decile_main_effects_{metric}"
    eff_dir.mkdir(parents=True, exist_ok=True)

    df = df_m[['subject_id', 'condition', 'decile', metric]].copy()

    df['modality'] = df['condition'].apply(lambda s: 'Physical' if s.endswith('P') else 'Digital')
    df['constraint'] = df['condition'].apply(lambda s: 'Open' if s.startswith('O') else 'Constrained')

    deciles = sorted(df['decile'].unique())

    rows_constraint, pvals_constraint = [], []
    rows_modality, pvals_modality = [], []

    plot_rows_constraint = []
    plot_rows_modality = []

    for d in deciles:
        dsub = df[df['decile'] == d]

        # Constrain main effects
        oc = (dsub.groupby(['subject_id', 'constraint'])[metric]
                    .mean()
                    .unstack('constraint'))  
        if {'Open', 'Constrained'}.issubset(set(oc.columns)):
            oc = oc[['Open', 'Constrained']].dropna()
            n_pairs = len(oc)
            if n_pairs >= 2:
                res = pg.ttest(oc['Open'], oc['Constrained'], paired=True).iloc[0]
                rows_constraint.append({
                    'metric': metric, 'decile': d, 'n_pairs': n_pairs,
                    'mean_Open': oc['Open'].mean(), 'mean_Constrained': oc['Constrained'].mean(),
                    'mean_diff(Open-Constrained)': oc['Open'].mean() - oc['Constrained'].mean(),
                    't': res['T'], 'dof': res['dof'], 'p': res['p-val'], 'q': np.nan,
                    'cohen_dz': res.get('cohen-d', np.nan)
                })
                pvals_constraint.append(res['p-val'])
            else:
                rows_constraint.append({
                    'metric': metric, 'decile': d, 'n_pairs': n_pairs,
                    'mean_Open': np.nan, 'mean_Constrained': np.nan,
                    'mean_diff(Open-Constrained)': np.nan,
                    't': np.nan, 'dof': np.nan, 'p': np.nan, 'q': np.nan, 'cohen_dz': np.nan
                })
                pvals_constraint.append(np.nan)

            tmp = oc.reset_index().melt(id_vars='subject_id', value_vars=['Open', 'Constrained'],
                                        var_name='level', value_name=metric)
            tmp['decile'] = d
            plot_rows_constraint.append(tmp)

        # Modality main effects
        pdm = (dsub.groupby(['subject_id', 'modality'])[metric]
                    .mean()
                    .unstack('modality'))
        if {'Physical', 'Digital'}.issubset(set(pdm.columns)):
            pdm = pdm[['Physical', 'Digital']].dropna()
            n_pairs = len(pdm)
            if n_pairs >= 2:
                res = pg.ttest(pdm['Physical'], pdm['Digital'], paired=True).iloc[0]
                rows_modality.append({
                    'metric': metric, 'decile': d, 'n_pairs': n_pairs,
                    'mean_Physical': pdm['Physical'].mean(), 'mean_Digital': pdm['Digital'].mean(),
                    'mean_diff(Physical-Digital)': pdm['Physical'].mean() - pdm['Digital'].mean(),
                    't': res['T'], 'dof': res['dof'], 'p': res['p-val'], 'q': np.nan,
                    'cohen_dz': res.get('cohen-d', np.nan)
                })
                pvals_modality.append(res['p-val'])
            else:
                rows_modality.append({
                    'metric': metric, 'decile': d, 'n_pairs': n_pairs,
                    'mean_Physical': np.nan, 'mean_Digital': np.nan,
                    'mean_diff(Physical-Digital)': np.nan,
                    't': np.nan, 'dof': np.nan, 'p': np.nan, 'q': np.nan, 'cohen_dz': np.nan
                })
                pvals_modality.append(np.nan)

            tmp2 = pdm.reset_index().melt(id_vars='subject_id', value_vars=['Physical', 'Digital'],
                                          var_name='level', value_name=metric)
            tmp2['decile'] = d
            plot_rows_modality.append(tmp2)


    df_con = pd.DataFrame(rows_constraint)
    df_mod = pd.DataFrame(rows_modality)
    con_csv = eff_dir / f"{metric}_main_effect_constraint_per_decile_ttests.csv"
    mod_csv = eff_dir / f"{metric}_main_effect_modality_per_decile_ttests.csv"
    df_con.to_csv(con_csv, index=False)
    df_mod.to_csv(mod_csv, index=False)
    print(f"Saved: {con_csv}")
    print(f"Saved: {mod_csv}")

    if len(plot_rows_constraint) > 0:
        plot_con = pd.concat(plot_rows_constraint, ignore_index=True)
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(data=plot_con, x='decile', y=metric, hue='level', style='level',
                          markers=True, dashes=False, errorbar='se')
        
        if not df_con.empty and 'p' in df_con.columns:
            sigs = df_con[(df_con['p'] < 0.05) & df_con['p'].notna()]
            if not sigs.empty:
                agg = (plot_con.groupby(['level', 'decile'], as_index=False)[metric]
                              .mean().rename(columns={metric: 'mean_val'}))
                for _, r in sigs.iterrows():
                    dec = int(r['decile'])
                    for lvl in ['Open', 'Constrained']:
                        y = agg[(agg['level'] == lvl) & (agg['decile'] == dec)]['mean_val']
                        if len(y) == 1 and pd.notnull(y.iloc[0]) and np.isfinite(y.iloc[0]):
                            ax.scatter(dec, y.iloc[0], facecolors='none', edgecolors='black',
                                       s=180, linewidths=2, zorder=6)
        ax.set_title(f"Main Effect (Constraint): Open vs Constrained — {metric.replace('_',' ').title()}")
        ax.set_xlabel("Time (Decile)")
        ax.set_ylabel(metric.replace('_',' ').title())
        ax.set_xticks(list(range(1, 11)))
        ax.legend(title='Constraint')
        plt.tight_layout()
        out_png = eff_dir / f"{metric}_main_effect_constraint_per_decile_plot.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved: {out_png}")

    if len(plot_rows_modality) > 0:
        plot_mod = pd.concat(plot_rows_modality, ignore_index=True)
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(data=plot_mod, x='decile', y=metric, hue='level', style='level',
                          markers=True, dashes=False, errorbar='se')
        if not df_mod.empty and 'p' in df_mod.columns:
            sigs = df_mod[(df_mod['p'] < 0.05) & df_mod['p'].notna()]
            if not sigs.empty:
                agg = (plot_mod.groupby(['level', 'decile'], as_index=False)[metric]
                              .mean().rename(columns={metric: 'mean_val'}))
                for _, r in sigs.iterrows():
                    dec = int(r['decile'])
                    for lvl in ['Physical', 'Digital']:
                        y = agg[(agg['level'] == lvl) & (agg['decile'] == dec)]['mean_val']
                        if len(y) == 1 and pd.notnull(y.iloc[0]) and np.isfinite(y.iloc[0]):
                            ax.scatter(dec, y.iloc[0], facecolors='none', edgecolors='black',
                                       s=180, linewidths=2, zorder=6)
        ax.set_title(f"Main Effect (Modality): Physical vs Digital — {metric.replace('_',' ').title()}")
        ax.set_xlabel("Time (Decile)")
        ax.set_ylabel(metric.replace('_',' ').title())
        ax.set_xticks(list(range(1, 11)))
        ax.legend(title='Modality')
        plt.tight_layout()
        out_png = eff_dir / f"{metric}_main_effect_modality_per_decile_plot.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved: {out_png}")

def statistical_analysis(out_dir: Path, df_metrics: pd.DataFrame):
    if df_metrics.empty:
        print("Metrics DataFrame is empty. Skipping analysis.")
        return

    df_metrics = _ensure_condition_factors(df_metrics)
    metrics_to_test = ['density', 'avg_degree_centrality', 'avg_clustering', 'global_efficiency']

    for metric in metrics_to_test:
        print(f"\n--- Analyzing: {metric.upper()} ---")
        df_avg = (df_metrics.groupby(['subject_id', 'modality', 'constraint'], as_index=False)[metric]
                            .mean())
        aov_overall = pg.rm_anova(data=df_avg, dv=metric, within=['modality', 'constraint'], subject='subject_id', detailed=True)
        aov_path = out_dir / f"anova_overall_{metric}.csv"
        aov_overall.to_csv(aov_path, index=False)
        print(f"Saved: {aov_path}")

        posthocs_overall_1 = pg.pairwise_tests(data=df_avg, dv=metric, within=['modality', 'constraint'], subject='subject_id')
        posthocs_overall_1.to_csv(out_dir / f"posthocs_overall_{metric}_1.csv", index=False)

        posthocs_overall_2 = pg.pairwise_tests(data=df_avg, dv=metric, within=['constraint', 'modality'], subject='subject_id')
        posthocs_overall_2.to_csv(out_dir / f"posthocs_overall_{metric}_2.csv", index=False)

        print(f"Saved: posthocs_overall_{metric}.csv") 
        
        cond_means = (df_avg.assign(conditions=df_avg['constraint'].str[0] + df_avg['modality'].str[0])
                              .groupby('conditions', as_index=False)[metric]
                              .mean()
                              .rename(columns={metric: 'mean_value'}))
        cond_means.to_csv(out_dir / f"means_overall_{metric}_by_condition.csv", index=False)

        constraint_means = (df_avg.groupby('constraint', as_index=False)[metric]
                                   .mean()
                                   .rename(columns={metric: 'mean_value'}))
        modality_means   = (df_avg.groupby('modality', as_index=False)[metric]
                                   .mean()
                                   .rename(columns={metric: 'mean_value'}))
        constraint_means.to_csv(out_dir / f"means_overall_{metric}_by_constraint.csv", index=False)
        modality_means.to_csv(out_dir / f"means_overall_{metric}_by_modality.csv", index=False)

        _pairs = [('OP','OD'),('OP','CP'),('CP','CD'),('OD','CD')]
        m = cond_means.set_index('conditions')['mean_value']
        rows = []
        for A,B in _pairs:
            if A in m and B in m:
                rows.append({'metric': metric, 'comparison': f'{A}_vs_{B}',
                             'mean_A': float(m[A]), 'mean_B': float(m[B]),
                             'mean_diff': float(m[A]-m[B])})
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"means_overall_{metric}_pairwise.csv", index=False)

        _paired_decile_ttests_for_metric(df_metrics[['subject_id','conditions','decile',metric]].copy(), metric, out_dir)
        _per_decile_main_effects(df_metrics[['subject_id','condition','decile',metric]].copy(), metric, out_dir)

def visualise_temporal_dynamics(out_dir: Path, df_metrics: pd.DataFrame, visualise=False):
    if not visualise:
        return
    if df_metrics.empty:
        return
    df_metrics = _ensure_condition_factors(df_metrics)
    sns.set_theme(style="whitegrid")
    metrics_to_test = ['density', 'avg_degree_centrality', 'avg_clustering', 'global_efficiency']
    for metric in metrics_to_test:
        plt.figure(figsize=(12, 7))
        ax = sns.lineplot(data=df_metrics, x='decile', y=metric, hue='conditions', style='conditions',
                          hue_order=condition_order, style_order=condition_order,
                          palette=condition_palette, markers=condition_markers,
                          dashes=False, errorbar='se')
        out_path = out_dir / f"{metric}_all_conditions.png"
        ax.set_title(f"Temporal Dynamics of {metric.replace('_', ' ').title()} (All Conditions)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (Decile)", fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xticks(range(1, 11))
        ax.legend(title='Condition')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

def run_connectivity_analysis(base_dir: Path, out_dir: Path):
    raw_list = load_data(base_dir)
    if not raw_list:
        print("No data loaded. Exiting.")
        return

    processed_raw_list = preprocessing(raw_list)
    decile_raw_list = create_deciles(processed_raw_list)
    all_connectivity_results = calc_corr_matrix(decile_raw_list)
    df_metrics = network_analysis(out_dir, all_connectivity_results)
    statistical_analysis(out_dir, df_metrics)
    visualise_temporal_dynamics(out_dir, df_metrics, visualise=True)


def main():
    parser = argparse.ArgumentParser(description="Process fNIRS data for connectivity analysis (paired t-tests)")
    parser.add_argument("--base-dir", "-b", type=Path,
                        default=Path("/Users/rashid/Library/CloudStorage/OneDrive-UniversityofBristol(2)/MSc Data Science/Dissertation/Technical Work/Objective 2/data/fnirs"),
                        help="Root folder containing all fNIRS raw data")
    parser.add_argument("--out-dir", "-o", type=Path,
                        default=Path("/Users/rashid/Library/CloudStorage/OneDrive-UniversityofBristol(2)/MSc Data Science/Dissertation/Technical Work/Objective 2/data/results/connectivity_analysis_ttests"),
                        help="Output directory for the processed connectivity analysis")
    args = parser.parse_args()
    run_connectivity_analysis(args.base_dir, args.out_dir)


if __name__ == "__main__":
    main()
