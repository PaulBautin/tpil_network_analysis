from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse de la connectivite avec commmit2
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
import bct
from functions.connectivity_read_files import find_files_with_common_name

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

def compute_degree(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.degrees_und(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_betweenness(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.betweenness_wei(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_eigenvector(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.eigenvector_centrality_und(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_cluster(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.clustering_coef_wu(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_shortest_path(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    inverted_matrix = 1 / np_connectivity_matrix
    inverted_matrix[np.isinf(inverted_matrix)] = 0 
    inverted_matrix[np.isnan(inverted_matrix)] = 0 
    D, B = bct.distance_wei(inverted_matrix)
    valid_values = D[~np.isin(D, [0,np.inf])] # Disconnected nodes resulting from filtering will result in inf length. Diagonal values have 0 length
    avg_shortest_path = np.mean(valid_values)
    return avg_shortest_path

def compute_global_efficiency(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    global_efficiency = bct.efficiency_wei(np_connectivity_matrix)
    return global_efficiency

def random_matrix(df_connectivity_matrix):
    # convert DataFrame to a NumPy array
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    # print number of edges and nodes of current graph
    num_edges = np.count_nonzero(np_connectivity_matrix)
    num_nodes = np_connectivity_matrix.shape[0]
    print("Number of edges (original graph):", num_edges)
    print("Number of nodes (original graph):", num_nodes)
    # Create random graph
    num_iter = 10 
    rand_result = bct.randmio_und(np_connectivity_matrix, num_iter)
    rand_graph = rand_result[0]
    np_rand_graph = np.array(rand_graph)
    num_edges_r = np.count_nonzero(np_rand_graph)
    num_nodes_r = np_rand_graph.shape[0]
    print("Number of edges (random graph):", num_edges_r)
    print("Number of nodes (random graph):", num_nodes_r)
    df_rand_connectivity_matrix = pd.DataFrame(np_rand_graph, index=df_connectivity_matrix.index, columns=df_connectivity_matrix.columns)
    return df_rand_connectivity_matrix

def compute_small_world(df_connectivity_matrix):
    df_rand_connectivity_matrix = random_matrix(df_connectivity_matrix)
    np_rand_connectivity_matrix = df_rand_connectivity_matrix.to_numpy()
    cluster_coeff = compute_cluster(df_connectivity_matrix)
    avg_cluster_coeff = np.mean(cluster_coeff)
    shortest_path = compute_shortest_path(df_connectivity_matrix)
    cluster_coeff_rand = compute_cluster(np_rand_connectivity_matrix)
    avg_cluster_coeff_rand = np.mean(cluster_coeff_rand)
    shortest_path_rand = compute_shortest_path(np_rand_connectivity_matrix)
    sigma = (avg_cluster_coeff / avg_cluster_coeff_rand) / (shortest_path / shortest_path_rand)
    
    return sigma

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
    
    ### Get connectivity data
    df_con = find_files_with_common_name(path_results_con, "commit2_weights.csv")
    df_clbp = find_files_with_common_name(path_results_clbp, "commit2_weights.csv")
    ### work only on one session at a time
    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    df_con_v2 = df_con[df_con['session'] == "v2"].drop("session", axis=1)
    df_clbp_v2 = df_clbp[df_clbp['session'] == "v2"].drop("session", axis=1)
    df_con_v3 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    df_clbp_v3 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)

    if __name__ == "__main__":
        main()