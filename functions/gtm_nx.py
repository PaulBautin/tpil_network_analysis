from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Fonctions pour l'analyse de la connectivité structurelle avec la théorie des graphes de nx
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
from functions.connectivity_read_files import find_files_with_common_name
from functions.connectivity_filtering import load_brainnetome_centroids
from functions.connectivity_filtering import distance_dependant_filter
from functions.connectivity_stats import mean_matrix, z_score


def networkx_graph_convertor(df_connectivity_matrix, df_weighted_nodes, metric):
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
    # Validate input
    valid_metrics = ['zscore', 'friedman']
    if metric not in valid_metrics:
        print("Invalid metric. Valid metrics:", valid_metrics)
        return None
    # specify graph parameters according to metric
    if metric == 'zscore':
        node_color = df_weighted_nodes
        node_color = (node_color.sort_index())
        print(node_color)
        node_color_values = node_color['centrality']
        vmin = -2
        vmax = 2
    if metric == 'friedman':
        node_color = df_weighted_nodes
        node_color = (node_color.sort_index())
        print(node_color)
        node_color_values = node_color['statistic']
        vmin = 0
        vmax = 6
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
    node_color = df_weighted_nodes
    node_color = (node_color.sort_index())
    print(node_color)
    nx.draw_networkx(G, pos=dict_coords, labels=dict_labels, node_color=node_color_values, cmap=colormap, vmin=vmin, vmax=vmax, with_labels=True)
    #set colorbar
    plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=colormap))
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
