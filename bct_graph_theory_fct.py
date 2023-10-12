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
from connectivity_read_files import find_files_with_common_name

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
    centrality = bct.distance_wei_floyd(df_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_global_efficiency(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    global_efficiency = bct.efficiency_wei(np_connectivity_matrix)
    return global_efficiency

def randomized_weighted_network(df_connectivity_matrix, num_iterations=1000):
    # convert DataFrame to a NumPy array
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    randomized_matrix = np_connectivity_matrix.copy()
    # Get the indices of non-zero elements (edges) in the original matrix
    non_zero_indices = np.transpose(np.nonzero(np_connectivity_matrix))
    # Perform edge rewiring
    for _ in range(num_iterations):
        # Randomly select two different edges to rewire
        i, j = non_zero_indices[np.random.choice(len(non_zero_indices), 2, replace=False)]

        # Swap the weights of the selected edges
        randomized_matrix[i, j], randomized_matrix[j, i] = randomized_matrix[j, i], randomized_matrix[i, j]

    return randomized_matrix

def compute_small_world(df_connectivity_matrix):
    rand_np_connectivity_matrix = df_connectivity_matrix.groupby('subject').apply(lambda x:randomized_weighted_network(x))
    cluster_coeff = df_connectivity_matrix.groupby('subject').apply(lambda x:compute_cluster(x))
    print(cluster_coeff)
    shortest_path = df_connectivity_matrix.groupby('subject').apply(lambda x:compute_shortest_path(x))
    print(shortest_path)
    cluster_coeff_rand = rand_np_connectivity_matrix.groupby('subject').apply(lambda x:compute_cluster(x))
    print(cluster_coeff_rand)
    shortest_path_rand = rand_np_connectivity_matrix.groupby('subject').apply(lambda x:compute_shortest_path(x))
    print(shortest_path_rand)
    sigma = (cluster_coeff / cluster_coeff_rand) / (shortest_path / shortest_path_rand)
    
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