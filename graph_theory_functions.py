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
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
import bct
import networkx as nx
from connectivity_read_files import find_files_with_common_name
from connectivity_filtering import load_brainnetome_centroids
from connectivity_filtering import distance_dependant_filter

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

def mean_matrix(df_connectivity_matrix):
    """
    Returns mean value of all subjects for a given edge

    Parameters
    ----------
    df_connectivity_matrix : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    
    Returns
    -------
    mean_matrix : (N, N) pandas DataFrame
    """
    mean_matrix = df_connectivity_matrix.groupby("roi").mean()  
    return mean_matrix

def z_score_centrality(df_con_v1, df_clbp_v1):
    """
    Returns mean z-score between clbp and con data for a given edge

    Parameters
    ----------
    df_con_v1 : (1,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clbp_v1 : (1,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    df_anor_mean : (1, N) pandas DataFrame
    """
    df_con_mean = mean_matrix(df_con_v1)
    df_con_std = df_con_v1.groupby("roi").std()
    df_anor = (df_clbp_v1 - df_con_mean) / df_con_std
    df_anor_mean = mean_matrix(df_anor)
    return df_anor_mean

def networkx_graph_convertor(df_connectivity_matrix, df_weighted_nodes):
    """
    Creates a networkx graph with nodes varying in color based on degree centrality of each nodes

    Parameters
    ----------
    df_connectivity_matrix : (N, N) pandas DataFrame that specifies the strength of edges between nodes
    df_weighted_nodes : (1, N) pandas DataFrame that specifies node weights based on degree centrality

    Returns
    -------
    Networkx graph
    """
    # read the labels from the .txt file
    with open('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all_atlas.txt', 'r') as labels_file:
        labels = [line.strip() for line in labels_file]
    # load the node centroids using the provided function
    coordinates_array = load_brainnetome_centroids()
    # convert DataFrame to a NumPy array
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    # create a colormap based on df_weighted_nodes
    colormap = plt.cm.get_cmap('RdYlBu')
    # create an empty graph using NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    # add nodes to the graph with labels from the .txt file and set their attributes
    dict_coords = {i: coordinates_array[i,:2] for i in range(len(labels))}
    dict_labels = {i: labels[i] for i in range(len(labels))}
    # add weighted edges to the graph based on the connectivity NumPy array
    num_nodes = len(labels)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # check if the weight is non-zero (i.e., there's a connection)
            weight = np_connectivity_matrix[i, j]
            if weight != 0:
                # add the weighted edge between nodes with the weight from the NumPy array
                G.add_edge(i, j, weight=weight)
    # graph according to node position and color intensity
    node_color = pd.read_csv('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/friedman_stat.csv')
    node_color = (node_color.sort_index())
    print(node_color)
    nx.draw_networkx(G, pos=dict_coords, labels=dict_labels, node_color=node_color['statistic'], cmap=colormap, vmin=0, vmax=6, with_labels=False) #for z-score, add vmin=-2,vmax=2
    #set colorbar
    plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=6), cmap=colormap)) #for z-score, add norm=mpl.colors.Normalize(vmin=-2, vmax=2),
    #cbar.set_label('Color Intensity')
    plt.axis('equal')
    plt.show()

def networkx_degree_centrality(df_connectivity_matrix):
    """
    Creates a networkx graph from sub-pl007_ses-v1__nativepro_seg_all_atlas.txt labels and load_brainnetome_centroids()
    coordinates. Calculates degree centrality of nodes and stores them in a pandas DataFrame.

    Parameters
    ----------
    df_connectivity_matrix : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    pd.DataFrame.from_dict(dict_centrality, orient='index') : (1, N) pandas DataFrame with centrality metrics for each labeled node N
    """
    # read the labels from the .txt file
    df_connectivity_matrix = mean_matrix(df_connectivity_matrix)
    with open('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all_atlas.txt', 'r') as labels_file:
        labels = [line.strip() for line in labels_file]
    # load the node centroids using the provided function
    coordinates_array = load_brainnetome_centroids()
    # convert DataFrame to a NumPy array
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    # create an empty graph using NetworkX
    G = nx.Graph()
    # add nodes to the graph with labels from the .txt file and set their attributes
    G.add_nodes_from(labels)
    dict_coords = {i: coordinates_array[i,:2] for i in range(len(labels))}
    dict_labels = {i: labels[i] for i in range(len(labels))}
    
    num_nodes = len(labels)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # check if the weight is non-zero (i.e., there's a connection)
            weight = np_connectivity_matrix[i, j]
            if weight != 0:
                # add the weighted edge between nodes with the weight from the NumPy array
                G.add_edge(labels[i], labels[j], weight=weight)
    dict_centrality = nx.degree_centrality(G)
    return pd.DataFrame.from_dict(dict_centrality, orient='index')

def networkx_betweenness_centrality(df_connectivity_matrix):
    """
    Creates a networkx graph from sub-pl007_ses-v1__nativepro_seg_all_atlas.txt labels and load_brainnetome_centroids()
    coordinates. Calculates betweenness centrality of nodes and stores them in a pandas DataFrame.

    Parameters
    ----------
    df_connectivity_matrix : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    pd.DataFrame.from_dict(dict_centrality, orient='index') : (1, N) pandas DataFrame with centrality metrics for each labeled node N
    """
    df_connectivity_matrix = mean_matrix(df_connectivity_matrix)
    df_inverse_connectivity_matrix = 1 / df_connectivity_matrix
    # convert DataFrame to a NumPy array
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    np_inverse_connectivity_matrix = df_inverse_connectivity_matrix.to_numpy()
    string_inverse_connectivity_matrix = np.array2string(np_inverse_connectivity_matrix)
    # read the labels from the .txt file
    with open('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all_atlas.txt', 'r') as labels_file:
        labels = [line.strip() for line in labels_file]
    # load the node centroids using the provided function
    coordinates_array = load_brainnetome_centroids()
    # create an empty graph using NetworkX
    G = nx.Graph()
    # add nodes to the graph with labels from the .txt file and set their attributes
    G.add_nodes_from(labels)
    dict_coords = {i: coordinates_array[i,:2] for i in range(len(labels))}
    dict_labels = {i: labels[i] for i in range(len(labels))}
    
    num_nodes = len(labels)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # check if the weight is non-zero (i.e., there's a connection)
            weight = np_connectivity_matrix[i, j]
            if weight != 0:
                # add the weighted edge between nodes with the weight from the NumPy array
                G.add_edge(labels[i], labels[j], weight=weight)
    dict_centrality = nx.betweenness_centrality(G, weight=string_inverse_connectivity_matrix)
    return pd.DataFrame.from_dict(dict_centrality, orient='index')

def networkx_eigenvector_centrality(df_connectivity_matrix):
    """
    Creates a networkx graph from sub-pl007_ses-v1__nativepro_seg_all_atlas.txt labels and load_brainnetome_centroids()
    coordinates. Calculates eigenvector centrality of nodes and stores them in a pandas DataFrame.

    Parameters
    ----------
    df_connectivity_matrix : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    pd.DataFrame.from_dict(dict_centrality, orient='index') : (1, N) pandas DataFrame with centrality metrics for each labeled node N
    """
    df_connectivity_matrix = mean_matrix(df_connectivity_matrix)
    # convert DataFrame to a NumPy array
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    string_connectivity_matrix = np.array2string(np_connectivity_matrix)
    # read the labels from the .txt file
    with open('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all_atlas.txt', 'r') as labels_file:
        labels = [line.strip() for line in labels_file]
    # load the node centroids using the provided function
    coordinates_array = load_brainnetome_centroids()
    # create an empty graph using NetworkX
    G = nx.Graph()
    # add nodes to the graph with labels from the .txt file and set their attributes
    G.add_nodes_from(labels)
    dict_coords = {i: coordinates_array[i,:2] for i in range(len(labels))}
    dict_labels = {i: labels[i] for i in range(len(labels))}
    
    num_nodes = len(labels)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # check if the weight is non-zero (i.e., there's a connection)
            weight = np_connectivity_matrix[i, j]
            if weight != 0:
                # add the weighted edge between nodes with the weight from the NumPy array
                G.add_edge(labels[i], labels[j], weight=weight)
    dict_centrality = nx.eigenvector_centrality(G, weight=string_connectivity_matrix)
    return pd.DataFrame.from_dict(dict_centrality, orient='index')

def networkx_cluster_coefficient(df_connectivity_matrix):
    """
    Creates a networkx graph from sub-pl007_ses-v1__nativepro_seg_all_atlas.txt labels and load_brainnetome_centroids()
    coordinates. Calculates cluster coefficient of nodes and stores them in a pandas DataFrame.

    Parameters
    ----------
    df_connectivity_matrix : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    pd.DataFrame.from_dict(dict_centrality, orient='index') : (1, N) pandas DataFrame with centrality metrics for each labeled node N
    """
    df_connectivity_matrix = mean_matrix(df_connectivity_matrix)
    # convert DataFrame to a NumPy array
    np_connectivity_matrix = df_connectivity_matrix.to_numpy()
    string_connectivity_matrix = np.array2string(np_connectivity_matrix)
    # read the labels from the .txt file
    with open('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all_atlas.txt', 'r') as labels_file:
        labels = [line.strip() for line in labels_file]
    # load the node centroids using the provided function
    coordinates_array = load_brainnetome_centroids()
    # create an empty graph using NetworkX
    G = nx.Graph()
    # add nodes to the graph with labels from the .txt file and set their attributes
    G.add_nodes_from(labels)
    dict_coords = {i: coordinates_array[i,:2] for i in range(len(labels))}
    dict_labels = {i: labels[i] for i in range(len(labels))}
    
    num_nodes = len(labels)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # check if the weight is non-zero (i.e., there's a connection)
            weight = np_connectivity_matrix[i, j]
            if weight != 0:
                # add the weighted edge between nodes with the weight from the NumPy array
                G.add_edge(labels[i], labels[j], weight=weight)
    dict_centrality = nx.clustering(G, weight=string_connectivity_matrix)
    return pd.DataFrame.from_dict(dict_centrality, orient='index')

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

    ### Get commit2_weights from all con subjects AND all clbp subjets
    df_con = find_files_with_common_name(path_results_con, "commit2_weights.csv")
    df_clbp = find_files_with_common_name(path_results_clbp, "commit2_weights.csv")

    ### Filter commit2_weights.csv to keep only the first visit (v1)
    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)

    ### Measure degree centrality of every node and store in a pandas DataFrame compatible with z_score_centrality
    df_con_centrality = df_con_v1.groupby('subject').apply(lambda x:networkx_betweenness_centrality(x)).rename(columns={0: 'centrality'})
    df_con_centrality.index.names = ['subject', 'roi']
    df_clbp_centrality = df_clbp_v1.groupby('subject').apply(lambda x:networkx_betweenness_centrality(x)).rename(columns={0: 'centrality'})
    df_clbp_centrality.index.names = ['subject', 'roi']
    ### Calculate the z-score for every node between clbp and con
    z_score_central = z_score_centrality(df_con_centrality, df_clbp_centrality)
    
    ### Calculate mean edge weight for clbp
    mean_clbp = mean_matrix(df_clbp_v1)
    
    ### Create networkx graph with varying colored nodes intensity based on z_score_central with edges determined by mean_clbp
    networkx_graph_convertor(mean_clbp, z_score_central)

if __name__ == "__main__":
    main()