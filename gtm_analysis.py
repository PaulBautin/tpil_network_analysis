from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour obtenir métriques de la théorie des graphes
#
# example: python connectivity_analysis.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Marc Antoine
#
# Prerequis: environnement virtuel avec python, pandas, numpy, netneurotools, scilpy et matplotlib (env_tpil)
#
#########################################################################################


# Parser
#########################################################################################


import pandas as pd
import numpy as np
import os
import argparse
import glob
import bct
import networkx as nx
from statsmodels.stats.anova import AnovaRM
from functions.connectivity_filtering import distance_dependant_filter, load_brainnetome_centroids, scilpy_filter, threshold_filter, sex_filter, pain_duration_filter
from functions.connectivity_processing import data_processor, prepare_data, difference_visits
from functions.connectivity_read_files import find_files_with_common_name
from functions.gtm_bct import  compute_betweenness, compute_cluster, compute_degree, compute_eigenvector, compute_efficiency, compute_small_world, compute_shortest_path, modularity_louvain, bct_master
from functions.gtm_nx import networkx_graph_convertor


def get_parser():
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Compute statistics based on the .csv files containing the tractometry metrics:",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-clbp",
        required=True,
        default='connectivity_results',
        help='Path to folder that contains output .csv files (e.g. "~/dev_tpil/tpil_network_analysis/data/22-11-16_connectoflow/clbp/sub-pl007_ses-v1/Compute_Connectivity")',
    )
    mandatory.add_argument(
        "-con",
        required=True,
        default='connectivity_results',
        help='Path to folder that contains output .csv files (e.g. "~/dev_tpil/tpil_network_analysis/data/22-11-16_connectoflow/control/sub-pl029_ses-v1/Compute_Connectivity")',
    )
    mandatory.add_argument(
        "-connectivity_type",
        required=True,
        default='connectivity_results',
        help='type of connectivity data to work with (e.g. "commit2_weights.csv")',
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-fig',
        help='Generate figures',
        action='store_true'
    )
    optional.add_argument(
        '-o',
        help='Path where figures will be saved. By default, they will be saved in the current directory.',
        default="."
    )
    return parser

def main():
    """
    main function, gather stats and call plots
    """
    ### Get parser elements
    parser = get_parser()
    arguments = parser.parse_args()
    path_results_con = os.path.abspath(os.path.expanduser(arguments.con))
    path_results_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    path_output = os.path.abspath(arguments.o)
    
    """ 
    Get connectivity data
    """
    df_con = find_files_with_common_name(path_results_con, arguments.connectivity_type)
    df_clbp = find_files_with_common_name(path_results_clbp, arguments.connectivity_type)
    ### work only on one session at a time
    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    df_con_v2 = df_con[df_con['session'] == "v2"].drop("session", axis=1)
    df_clbp_v2 = df_clbp[df_clbp['session'] == "v2"].drop("session", axis=1)
    df_con_v3 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    df_clbp_v3 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)
    
    
    """
    Calculate chosen graph theory metric for all subjects in the DataFrame
    """
    """
    Prepare data for analysis
    """
    ### Select data and filters to apply
    df_clean_clbp_v1 = data_processor(df_clbp_v1, session='all', condition='clbp', filter='scilpy', clean=True)
    df_clean_con_v1 = data_processor(df_con_v1, session='all', condition='con', filter='scilpy', clean=True)
    df_clean_clbp_v2 = data_processor(df_clbp_v2, session='all', condition='clbp', filter='scilpy', clean=True)
    df_clean_con_v2 = data_processor(df_con_v2, session='all', condition='con', filter='scilpy', clean=True)
    df_clean_clbp_v3 = data_processor(df_clbp_v3, session='all', condition='clbp', filter='scilpy', clean=True)
    df_clean_con_v3 = data_processor(df_con_v3, session='all', condition='con', filter='scilpy', clean=True)

    degree_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'degree_centrality'))
    degree_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'degree_centrality'))
    degree_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'degree_centrality'))
    degree_v1.insert(0, "visit", 1, True)
    degree_v2.insert(0, "visit", 2, True)
    degree_v3.insert(0, "visit", 3, True)
    df_degree = pd.concat([degree_v1, degree_v2, degree_v3])

    strength_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality', method='local'))
    strength_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality', method='local'))
    strength_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality', method='local'))
    strength_v1.insert(0, "visit", 1, True)
    strength_v2.insert(0, "visit", 2, True)
    strength_v3.insert(0, "visit", 3, True)
    df_strength = pd.concat([strength_v1, strength_v2, strength_v3])

    betweenness_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'betweenness_centrality'))
    betweenness_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'betweenness_centrality'))
    betweenness_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'betweenness_centrality'))
    betweenness_v1.insert(0, "visit", 1, True)
    betweenness_v2.insert(0, "visit", 2, True)
    betweenness_v3.insert(0, "visit", 3, True)
    df_betweenness = pd.concat([betweenness_v1, betweenness_v2, betweenness_v3])

    cluster_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'cluster_coefficient', method='local'))
    cluster_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'cluster_coefficient', method='local'))
    cluster_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'cluster_coefficient', method='local'))
    cluster_v1.insert(0, "visit", 1, True)
    cluster_v2.insert(0, "visit", 2, True)
    cluster_v3.insert(0, "visit", 3, True)
    df_cluster = pd.concat([cluster_v1, cluster_v2, cluster_v3])

    efficiency_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'efficiency', method='local'))
    efficiency_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'efficiency', method='local'))
    efficiency_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'efficiency', method='local'))
    efficiency_v1.insert(0, "visit", 1, True)
    efficiency_v2.insert(0, "visit", 2, True)
    efficiency_v3.insert(0, "visit", 3, True)
    df_efficiency = pd.concat([efficiency_v1, efficiency_v2, efficiency_v3])
   
    strength_col = df_strength['metric']
    betweenness_col = df_betweenness['metric']
    cluster_col = df_cluster['metric']
    efficiency_col = df_efficiency['metric']
    degree = df_degree.rename(columns={"metric": "degree"})
    degree['strength'] = strength_col
    degree['betweenness'] = betweenness_col
    degree['cluster'] = cluster_col
    degree['efficiency'] = efficiency_col
    degree.to_csv('/Users/Marc-Antoine/Documents/tpil_network_analysis/results/gtm_metrics_con.csv')
    
    """
    Calculate chosen global graph theory metric for all subjects in the DataFrame
    """
    """
    Prepare data for analysis
    """
    ### Select data and filters to apply
    # df_clean_clbp_v1 = data_processor(df_clbp_v1, session='all', condition='clbp', filter='threshold', clean=True)
    # df_clean_con_v1 = data_processor(df_con_v1, session='all', condition='con', filter='threshold', clean=True)
    # df_clean_clbp_v2 = data_processor(df_clbp_v2, session='all', condition='clbp', filter='threshold', clean=True)
    # df_clean_con_v2 = data_processor(df_con_v2, session='all', condition='con', filter='threshold', clean=True)
    # df_clean_clbp_v3 = data_processor(df_clbp_v3, session='all', condition='clbp', filter='threshold', clean=True)
    # df_clean_con_v3 = data_processor(df_con_v3, session='all', condition='con', filter='threshold', clean=True)
    # efficiency_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'efficiency', method='global'))
    # efficiency_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'efficiency', method='global'))
    # efficiency_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'efficiency', method='global'))
    # efficiency_v1.insert(0, "visit", 1, True)
    # efficiency_v2.insert(0, "visit", 2, True)
    # efficiency_v3.insert(0, "visit", 3, True)
    # df_efficiency = pd.concat([efficiency_v1, efficiency_v2, efficiency_v3])
    # df_efficiency.index = df_efficiency.index.droplevel(1)
    # df_efficiency.drop_duplicates(inplace=True)
    # df_efficiency.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/efficiency_con.csv')

    # cluster_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'cluster_coefficient', method='global'))
    # cluster_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'cluster_coefficient', method='global'))
    # cluster_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'cluster_coefficient', method='global'))
    # cluster_v1.insert(0, "visit", 1, True)
    # cluster_v2.insert(0, "visit", 2, True)
    # cluster_v3.insert(0, "visit", 3, True)
    # df_cluster = pd.concat([cluster_v1, cluster_v2, cluster_v3])
    # df_cluster.index = df_cluster.index.droplevel(1)
    # df_cluster.drop_duplicates(inplace=True)
    # df_cluster.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/cluster_con.csv')

    # strength_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality', method='global'))
    # strength_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality', method='global'))
    # strength_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'strength_centrality', method='global'))
    # strength_v1.insert(0, "visit", 1, True)
    # strength_v2.insert(0, "visit", 2, True)
    # strength_v3.insert(0, "visit", 3, True)
    # df_strength = pd.concat([strength_v1, strength_v2, strength_v3])
    # df_strength.index = df_strength.index.droplevel(1)
    # df_strength.drop_duplicates(inplace=True)
    # df_strength.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/strength_con.csv')
    
    # small_world_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'small_world'))
    # small_world_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'small_world'))
    # small_world_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'small_world'))
    # small_world_v1.insert(0, "visit", 1, True)
    # small_world_v2.insert(0, "visit", 2, True)
    # small_world_v3.insert(0, "visit", 3, True)
    # df_small_world = pd.concat([small_world_v1, small_world_v2, small_world_v3])
    # df_small_world.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/small_world_con.csv')
    
    # modularity_v1 = df_clean_con_v1.groupby('subject').apply(lambda x:bct_master(x, 'modularity'))
    # modularity_v2 = df_clean_con_v2.groupby('subject').apply(lambda x:bct_master(x, 'modularity'))
    # modularity_v3 = df_clean_con_v3.groupby('subject').apply(lambda x:bct_master(x, 'modularity'))
    # modularity_v1.insert(0, "visit", 1, True)
    # modularity_v2.insert(0, "visit", 2, True)
    # modularity_v3.insert(0, "visit", 3, True)
    # df_modularity = pd.concat([modularity_v1, modularity_v2, modularity_v3])
    # df_modularity.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/modularity_con.csv')

    # strength = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/strength_con.csv')
    # strength_col = strength['metric']
    # cluster = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/cluster_con.csv')
    # cluster_col = cluster['metric']
    # small_world = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/small_world_con.csv')
    # small_world_col = small_world['metric']
    # modularity = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/modularity_con.csv')
    # modularity_col = modularity['metric']

    # efficiency = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global/efficiency_con.csv')
    # df_efficiency = efficiency.rename(columns={"metric": "efficiency"})
    # df_efficiency['strength'] = strength_col
    # df_efficiency['cluster'] = cluster_col
    # df_efficiency['small_world'] = small_world_col
    # df_efficiency['modularity'] = modularity_col
    # print(df_efficiency)
    
    # df_efficiency.to_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/gtm_global_metrics_con.csv')


if __name__ == "__main__":
    main()