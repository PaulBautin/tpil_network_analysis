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
# Prerequis: environnement virtuel avec python, pandas, numpy et matplotlib (env_tpil)
#
#########################################################################################


# Parser
#########################################################################################


import pandas as pd
import numpy as np
import os
import networkx as nx
import argparse
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob


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
    mean_matrix = df_connectivity_matrix.groupby("roi").mean()
    return mean_matrix

def z_score(df_con_v1, df_clbp_v1):
    df_con_mean = mean_matrix(df_con_v1)
    df_con_std = df_con_v1.groupby("roi").std()
    df_clbp = df_clbp_v1.set_index(["subject","roi"])
    df_anor = (df_clbp - df_con_mean) / df_con_std
    df_anor_mean = mean_matrix(df_anor)
    return df_anor_mean

#def graph_matrix(df_connectivity_matrix):
    G = nx.Graph(df_connectivity_matrix)
    num_nodes = df_connectivity_matrix.shape[0]
    numbers = [f"{i+1:03d}" for i in range(num_nodes)]
    names = []
    with open('/home/mafor/dev_tpil/tpil_network_analysis/data/Brainnetome atlas.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            names.append(x)
    
    nx.set_node_attributes(G, names, 'label')
    pos = nx.circular_layout(G)
    graph = nx.draw_networkx(G, pos=pos, with_labels=True, node_color='skyblue', node_size=50, font_size=10, edge_color='gray')
    #node_label_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}  # Adjust label position
    #nx.draw_networkx_labels(G, node_label_pos, labels=names)
    return graph

def binary_mask(df_connectivity_matrix):
    df_abs_matrix = np.abs(df_connectivity_matrix)
    df_binary_matrix = (df_abs_matrix > 0).astype(np.int_)
    #np.where(df_abs_matrix > threshold, upper, lower)
    df_upper_binary_matrix = np.triu(df_binary_matrix)
    return df_upper_binary_matrix


def circle(df_connectivity_matrix):
    A = df_connectivity_matrix.values  # Convert DataFrame to NumPy array
    N = A.shape[0]
    # x/y coordinates of nodes in a circular layout
    r = 1
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    # labels of nodes
    numbers = [f"{i+1:03d}" for i in range(N)]
    names = []
    with open('/home/mafor/dev_tpil/tpil_network_analysis/data/Brainnetome atlas.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            names.append(x)
    txt = [f"{name}:{number}" for name, number in zip(names, numbers)]
    # seaborn styling
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 8))
    # show nodes and edges
    line_plot = plt.plot(xy[:, 0], xy[:, 1], linestyle="none", marker="o", markersize=10, color="steelblue", alpha=0.7)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] != 0:  # Change condition to consider non-zero edge intensities
                edge_color = A[i, j]  # Use the edge intensity as the color value
                plt.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=cm.viridis(edge_color), linewidth=1)
                label_x_i = xy[i, 0] * 1.22  # Adjust label positioning for node i
                label_y_i = xy[i, 1] * 1.22  # Adjust label positioning for node i
                rotation_i = theta[i] * 180 / np.pi
                label_x_j = xy[j, 0] * 1.22  # Adjust label positioning for node j
                label_y_j = xy[j, 1] * 1.22  # Adjust label positioning for node j
                rotation_j = theta[j] * 180 / np.pi
                if theta[i] > np.pi / 2 and theta[i] < 3 * np.pi / 2:
                    rotation_i += 180  # Rotate labels on the left side of the circle by 180 degrees
                if theta[j] > np.pi / 2 and theta[j] < 3 * np.pi / 2:
                    rotation_j += 180  # Rotate labels on the left side of the circle by 180 degrees
                plt.text(label_x_i, label_y_i, txt[i], fontsize=8, rotation=rotation_i, ha='center', va='center')
                plt.text(label_x_j, label_y_j, txt[j], fontsize=8, rotation=rotation_j, ha='center', va='center')
    plt.axis([-1, 1, -1, 1])
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()

    # Create a colorbar for edge intensities
    sm = cm.ScalarMappable(cmap=cm.viridis)
    sm.set_array(A.flatten())
    cbar = plt.colorbar(sm)
    cbar.set_label('Edge Intensity')
    
    return line_plot

def find_files_with_common_name(directory, common_name):

    file_paths = glob.glob(directory + '/*/Compute_Connectivity/' + common_name)
    n = range(len(file_paths))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(file_paths[i]))) : pd.read_csv(file_paths[i], header=None) for i in n}
    # Remove last 3 columns and rows from each matrix
    for key in dict_paths:
        dict_paths[key] = dict_paths[key].iloc[:-3, :-3]
    df_paths = pd.concat(dict_paths)
    df_paths = df_paths.reset_index().rename(columns={'level_0': 'participant_id', 'level_1': 'roi'})
    # df_paths = df_paths[df_paths['participant_id'].str.contains('_ses-v1')]
    df_paths[['subject', 'session']] = df_paths['participant_id'].str.rsplit('_ses-', 1, expand=True)
    df_paths = df_paths.drop("participant_id", axis=1)
    return df_paths

def filter_no_connections(df_connectivity_matrix):
    #df_connectivity_matrix_numeric = df_connectivity_matrix.copy()
    #df_connectivity_matrix_numeric.iloc[:, 1:] = df_connectivity_matrix.iloc[:, 1:].replace(r'^\D+$', np.nan, regex=True)
    df_connectivity_matrix.iloc[:, 1:] = df_connectivity_matrix.iloc[:, 1:].astype(float)
    df_zero_connections = []
    df_zero_matrix = np.zeros_like(df_connectivity_matrix.iloc[:, 1:])
    for row in range(df_zero_matrix.shape[0]):
        for col in range(df_zero_matrix.shape[1]):
            if df_connectivity_matrix.iloc[row, col+1] < 1:
                df_zero_matrix[row, col] = 0
                df_zero_connections.append((row, col))
            else:
                df_zero_matrix[row, col] = 1
    #If mean of ROI < 1, has at least one 0
    df_mean_matrix = np.mean(df_zero_matrix.reshape(-1, 246, 246), axis=0)
    df_mean_matrix[df_mean_matrix < 1] = 0
    return df_mean_matrix

def histogram(df_connectivity_matrix):
    data = df_connectivity_matrix.values.flatten()
    data_nonzero = data[data != 0]
    percentiles = np.arange(0, 100, 5)
    bin_edges = np.percentile(data_nonzero, percentiles)
    hist, bins = np.histogram(data_nonzero, bins=bin_edges)
    plt.bar(bins[:-1], hist, width=bins[1]-bins[0])
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title('Histogram of Connectivity Matrix')
    
    return hist, bins
 
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

    df_con = find_files_with_common_name(path_results_con, "commit2_weights.csv")
    df_clbp = find_files_with_common_name(path_results_clbp, "commit2_weights.csv")

    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    
    df_con_sc = find_files_with_common_name(path_results_con, "sc.csv")
    df_clbp_sc = find_files_with_common_name(path_results_con, "sc.csv")
    
    df_con_sc_v1 = df_con[df_con_sc['session'] == "v1"].drop(["subject", "session"], axis=1)
    df_clbp_sc_v1 = df_clbp[df_clbp_sc['session'] == "v1"].drop(["subject", "session"], axis=1)
    
    df_z_score_v1 = z_score(df_con_v1, df_clbp_v1) 
    df_z_score_v1[filter_no_connections(df_con_sc_v1) == 0] = 0
    df_z_score_v1[filter_no_connections(df_clbp_sc_v1) == 0] = 0
    #df_z_score_hist = histogram(df_z_score_v1)
    #print(df_z_score_hist)
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/z_score.csv', df_z_score_v1_binary, fmt='%1.3f')
    #df_graph_z_score_v1 = circle(df_z_score_v1)
    #plt.show()

    #plt.imshow(df_z_score_v1, cmap='bwr', norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=20))
    #plt.colorbar()
    #plt.show()
    
    df_con_mean = mean_matrix(df_con_v1)
    df_con_mean[filter_no_connections(df_con_sc_v1) == 0] = 0
    df_con_mean[filter_no_connections(df_clbp_sc_v1) == 0] = 0
    df_con_hist = histogram(df_con_mean)
    print(df_con_hist)
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/con_mean_hist.txt', df_con_mean, fmt='%1.3f')
    df_con_binary = binary_mask(df_con_mean)
    #df_con_graph = circle(df_con_binary)
    #plt.show()

    df_clbp_mean = mean_matrix(df_clbp_v1)
    df_clbp_mean[filter_no_connections(df_clbp_sc_v1) == 0] = 0
    df_clbp_mean[filter_no_connections(df_clbp_sc_v1) == 0] = 0
    df_clbp_hist = histogram(df_clbp_mean)
    print(df_clbp_hist)
    #plt.show()
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/clbp_mean_hist.txt', df_clbp_mean, fmt='%1.3f')
    df_clbp_binary = binary_mask(df_clbp_mean)
    #df_graph_clbp = circle(df_clbp_binary)
    #plt.show()
    
if __name__ == "__main__":
    main()