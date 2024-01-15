from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Fonctions pour l'analyse de la connectivité structurelle avec la théorie des graphes de bct
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
import bct

def bct_master(df_connectivity_matrix, metric_name, **kwargs):
    # Permitted metrics for analysis
    metric_fcts = {
        'degree_centrality': compute_degree,
        'strength_centrality': compute_strength,
        'betweenness_centrality': compute_betweenness,
        'eigenvector_centrality': compute_eigenvector,
        'cluster_coefficient': compute_cluster,
        'shortest_path': compute_shortest_path,
        'global_efficiency': compute_global_efficiency,
        'small_world': compute_small_world,
        'modularity': modularity_louvain
    }
    # Check if the specified metric is in the dictionary
    if metric_name in metric_fcts:
        # Call the corresponding metric function with the connectivity matrix and additional parameters
        metric_fct = metric_fcts[metric_name]
        result = metric_fct(df_connectivity_matrix, **kwargs)
        return result
    else:
        print("Invalid metric name. Supported metrics:", list(metric_fcts.keys()))
        return None

def compute_degree(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.degrees_und(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, index=df_connectivity_matrix.index, columns=['centrality'])
    return df_centrality

def compute_strength(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.strengths_und(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, index=df_connectivity_matrix.index, columns=['centrality'])
    return df_centrality

def compute_betweenness(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.betweenness_wei(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, index=df_connectivity_matrix.index, columns=['centrality'])
    return df_centrality

def compute_eigenvector(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    centrality = bct.eigenvector_centrality_und(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, index=df_connectivity_matrix.index, columns=['centrality'])
    return df_centrality

def compute_cluster(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    np_connectivity_matrix_nrm = bct.weight_conversion(np_connectivity_matrix, 'normalize')
    centrality = bct.clustering_coef_wu(np_connectivity_matrix_nrm)
    df_centrality = pd.DataFrame(centrality, index=df_connectivity_matrix.index, columns=['centrality'])
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
    # np_rand_connectivity_matrix = df_rand_connectivity_matrix.to_numpy()
    cluster_coeff = compute_cluster(df_connectivity_matrix)
    avg_cluster_coeff = np.mean(cluster_coeff)
    shortest_path = compute_shortest_path(df_connectivity_matrix)
    cluster_coeff_rand = compute_cluster(df_rand_connectivity_matrix)
    avg_cluster_coeff_rand = np.mean(cluster_coeff_rand)
    shortest_path_rand = compute_shortest_path(df_rand_connectivity_matrix)
    sigma = (avg_cluster_coeff / avg_cluster_coeff_rand) / (shortest_path / shortest_path_rand)
    
    return sigma

def modularity_louvain(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    A, B = bct.community_louvain(np_connectivity_matrix)
    df_centrality = pd.DataFrame(A, index=df_connectivity_matrix.index)
    return A, B
