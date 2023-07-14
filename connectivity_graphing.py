from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# script pour l'analyse de la connectivite avec commmit2
#
# example: python connectivity_analysis.py -clbp <dir> -con <dir>
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
import networkx as nx
import argparse
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
from connectivity_read_files import find_files_with_common_name
from netneurotools.utils import get_centroids
from netneurotools.plotting import plot_point_brain




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

def circle_graph(np_connectivity_matrix):
    """
    Plots a circular graph of nodes with edge intensity and brainnetome labels

    Parameters
    ----------
    np_connectivity_matrix : (N,N) array_like

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    np_connectivity_matrix[np.isnan(np_connectivity_matrix)] = 0
    A = np_connectivity_matrix
    N = A.shape[0] #length of matrix
    # x/y coordinates of nodes in a circular layout
    r = 1
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    # labels of nodes with Brainnetome atlas
    numbers = [f"{i+1:03d}" for i in range(N)]
    names = []
    with open('/home/mafor/dev_tpil/tpil_network_analysis/labels/Brainnetome atlas.txt', 'r') as fp: 
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
            if A[i, j] != 0:  # Considers all non-zero edge intensities
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
    plt.show()

def histogram(np_connectivity_matrix):
    """
    Plots a histogram of connectivity matrix

    Parameters
    ----------
    np_connectivity_matrix : (N,N) array_like

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    data = np_connectivity_matrix.flatten()
    data[np.isnan(data)] = 0
    data_nonzero = data[data != 0]
    percentiles = np.arange(0, 100, 5) # each bin contains 5% of all data
    bin_edges = np.percentile(data_nonzero, percentiles)
    hist, bins = np.histogram(data_nonzero, bins=bin_edges)

    # Generate a list of colors for the bars using the "Pastel1" colormap
    num_bins = len(bins) - 1
    colors = plt.cm.Pastel1(np.linspace(0, 1, num_bins))

    # Plotting the histogram with colored bars
    fig, ax = plt.subplots()
    for i in range(num_bins):
        ax.bar(bins[i], hist[i], width=bins[i+1]-bins[i], color=colors[i])

    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Connectivity Matrix')

    # Add ticks to show bin edges
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{bin:.2f}' for bin in bins])
    
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

    #circular_graph = circle_graph(df_con_v1)
    #hist_graph = histogram(df_con_v1)


if __name__ == "__main__":
    main()