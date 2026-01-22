#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import ttest_rel, shapiro, wilcoxon

def load_data(base_dir, condition=None, ruleset=None):
    all_dsm_csvs = os.listdir(base_dir)
    all_dsm_paths = [os.path.join(base_dir, f) for f in all_dsm_csvs if f.endswith('.csv')]
    if condition !=None:
        all_dsm_paths = [f for f in all_dsm_paths if condition in f]
    else:
        all_dsm_paths = all_dsm_paths
    
    if ruleset != None:
        all_dsm_paths = [f for f in all_dsm_paths if ruleset in f]
    else:
        all_dsm_paths = all_dsm_paths
    
    all_dsm_dfs = [pd.read_csv(f, index_col=0, header=0) for f in all_dsm_paths]

    return all_dsm_paths, all_dsm_dfs

def graph_transformed_data(dsm_dfs):
    graphs = [nx.from_numpy_array(dsm.values, create_using=nx.Graph) for dsm in dsm_dfs]

    return graphs

def filename_parse(filepath):
    basename = os.path.basename(filepath)
    match = re.match(r'CON_P(\d+)_([A-Z]+)_R(\d+)\.csv', basename, re.IGNORECASE)
    if match:
        return {
            'participant': int(match.group(1)),
            'condition': match.group(2).upper(), 
            'ruleset': int(match.group(3))
        }
    print(f"Warning: Filename '{basename}' did not match expected pattern 'CON_P#_CONDITION_R#.csv'.")
    return None

def graph_metrics(all_dsm_paths, graphs):
    metrics = []
    for i, G in enumerate(graphs):
        parsed_info = filename_parse(all_dsm_paths[i])
        if not parsed_info:
            print(f"Skipping file due to parsing error: {all_dsm_paths[i]}")
            continue

        metric_data = {'filepath': all_dsm_paths[i]}
        metric_data.update(parsed_info)

        num_nodes = G.number_of_nodes()
        if num_nodes == 0:
            metric_data.update({
                'density': 0, 'avg_degree': 0, 'avg_betweenness': 0, 
                'avg_closeness': 0, 'global_efficiency': 0
            })
        else:
            metric_data.update({
                'density': nx.density(G),
                'avg_degree': sum(nx.degree_centrality(G).values()) / num_nodes,
                'avg_betweenness': sum(nx.betweenness_centrality(G, normalized=True).values()) / num_nodes,
                'avg_closeness': sum(nx.closeness_centrality(G).values()) / num_nodes,
                'global_efficiency': nx.global_efficiency(G)
            })
        
        metrics.append(metric_data)

    return pd.DataFrame(metrics)

def aggregated_metrics(metrics_df, group_by=None):
    if group_by not in metrics_df.columns:
        raise ValueError(f"Group by column '{group_by}' not found in DataFrame.")
    
    agg_cols = {
        'density': 'mean', 'avg_degree': 'mean', 'avg_betweenness': 'mean',
        'avg_closeness': 'mean', 'global_efficiency': 'mean',
    }
    
    aggregated = metrics_df.groupby(group_by).agg(agg_cols).reset_index()
    return aggregated

def collect_results(wide_df, metric, pairs, comparison_type, results):
    for a, b in pairs:
        if a not in wide_df.columns or b not in wide_df.columns:
            results.append({
                "Metric": metric,
                "Comparison": f"{a} vs {b}",
                "Type": comparison_type,
                "Test": None,
                "Stat": None,
                "p": None,
                "Mean_A": None,
                "Mean_B": None,
                "Mean_Diff": None,
                "N": 0,
                "Flag": "missing"
            })
            continue

        sub = wide_df[[a, b]].dropna()
        if sub.empty:
            results.append({
                "Metric": metric,
                "Comparison": f"{a} vs {b}",
                "Type": comparison_type,
                "Test": None,
                "Stat": None,
                "p": None,
                "Mean_A": None,
                "Mean_B": None,
                "Mean_Diff": None,
                "N": 0,
                "Flag": "no_overlap"
            })
            continue

        mean_a = float(sub[a].mean())
        mean_b = float(sub[b].mean())
        mean_diff = float((sub[a] - sub[b]).mean())
        n = int(len(sub))

        diff = sub[a] - sub[b]
        shapiro_p = shapiro(diff)[1] if len(diff) >= 3 else 1.0

        if shapiro_p > 0.05:
            stat, p = ttest_rel(sub[a], sub[b], nan_policy="omit")
            test_type = "t-test"
        else:
            try:
                stat, p = wilcoxon(sub[a], sub[b])
                test_type = "Wilcoxon"
            except ValueError:
                stat, p = None, None
                test_type = "Wilcoxon (error)"

        flag = "ns" if p is None or p > 0.05 else "sig"

        results.append({
            "Metric": metric,
            "Comparison": f"{a} vs {b}",
            "Type": comparison_type,
            "Test": test_type,
            "Stat": stat,
            "p": p,
            "Mean_A": mean_a,
            "Mean_B": mean_b,
            "Mean_Diff": mean_diff,
            "N": n,
            "Flag": flag
        })


def pivot(df, metric, pivot_by):
    return df.pivot_table(
        index="participant",
        columns=f"{pivot_by}",
        values=metric,
        aggfunc="mean"   
    )

def make_main_effect_wide(metrics_df, metric):

    wide = pivot(metrics_df, metric, "condition") 

    out = pd.DataFrame(index=wide.index)
    out['O'] = wide[['OP', 'OD']].mean(axis=1, skipna=True)
    out['C'] = wide[['CP', 'CD']].mean(axis=1, skipna=True)
    out['P'] = wide[['OP', 'CP']].mean(axis=1, skipna=True)
    out['D'] = wide[['OD', 'CD']].mean(axis=1, skipna=True)
    return out

def run_statistical_analysis(base_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_dsm_paths, all_dsm_dfs = load_data(base_dir)
    graphs = graph_transformed_data(all_dsm_dfs)

    #Individual Stats
    metrics_df = graph_metrics(all_dsm_paths, graphs)
    metrics_df.to_csv(os.path.join(out_dir, 'spaceship_metrics.csv'), index=False)

    #Aggregated Stats
    aggregated_conditions = aggregated_metrics(metrics_df, group_by='condition')
    aggregated_rulesets = aggregated_metrics(metrics_df, group_by='ruleset')
    aggregated_participants = aggregated_metrics(metrics_df, group_by='participant')

    aggregated_conditions.to_csv(os.path.join(out_dir, 'aggregated_conditions.csv'), index=False)
    aggregated_rulesets.to_csv(os.path.join(out_dir, 'aggregated_rulesets.csv'), index=False)
    aggregated_participants.to_csv(os.path.join(out_dir, 'aggregated_participants.csv'), index=False)

    #Comparison Stats
    pairs_condition_p_d = [("OP", "OD"), ("CP", "CD")]
    pairs_condition_o_c = [("OP", "CP"), ("OD", "CD")]
    pairs_rulesets = [(1, 2), (1, 4), (1, 5), (2, 4), (2, 5), (4, 5)]

    metrics_to_test = ['density', 'avg_degree', 'avg_betweenness', 'avg_closeness', 'global_efficiency']

    results_condition = []
    results_ruleset = []

    for metric in metrics_to_test:
        metric_title = metric.replace('_', ' ').title()
        condition_pivot_df = pivot(metrics_df, metric, "condition")
        collect_results(condition_pivot_df, metric_title, pairs_condition_o_c, "Open vs Constrained", results_condition)
        collect_results(condition_pivot_df, metric_title, pairs_condition_p_d, "Physical vs Digital", results_condition)
    
        main_effect_wide = make_main_effect_wide(metrics_df, metric)
        collect_results(main_effect_wide, metric_title, pairs=[("O", "C")], comparison_type="Main Effect: Constraint (O vs C)", results=results_condition)
        collect_results(main_effect_wide, metric_title, pairs=[("P", "D")], comparison_type="Main Effect: Modality (P vs D)", results=results_condition)

        ruleset_pivot_df = pivot(metrics_df, metric, "ruleset")
        collect_results(ruleset_pivot_df, metric_title, pairs_rulesets, "Ruleset Comparison", results_ruleset)

    results_condition_df = pd.DataFrame(results_condition)
    results_ruleset_df = pd.DataFrame(results_ruleset)

    output_path_ruleset = os.path.join(out_dir, "ruleset_comparison_stats_results.csv")
    results_ruleset_df.to_csv(output_path_ruleset, index=False)
    print(f"Ruleset Comparison Results saved to {output_path_ruleset}")

    output_path_condition = os.path.join(out_dir, "condition_comparison_stats_results.csv")
    results_condition_df.to_csv(output_path_condition, index=False)
    print(f"Condition Comparison Results saved to {output_path_condition}")

    print("All files saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Process and compare individual stats from DSM data.")
    parser.add_argument(
        "--base-dir", "-b",
        type=Path, 
        default=Path("./data/labelled_dsm_data/CONM"),
        help="Base directory containing DSM data.")
    
    parser.add_argument(
        "--out-dir", "-o", 
        type=Path, 
        default=Path("./data/results/matrix_comparisons"),
        help="Output directory for results.")
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    out_dir = args.out_dir
    
    run_statistical_analysis(base_dir, out_dir)

if __name__ == "__main__":
    main()


