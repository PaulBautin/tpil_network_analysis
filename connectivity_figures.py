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
import argparse
import seaborn as sns
import matplotlib
from netneurotools.utils import get_centroids

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
from connectivity_read_files import find_files_with_common_name
from netneurotools.plotting import plot_point_brain
from netneurotools.metrics import degrees_und
from connectivity_filtering import load_brainnetome_centroids


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

def connectivity_matrix_viewer(conn_matrix):
    """
    Prints connectivity_matrix

    Parameters
    ----------
    conn_matrix: (N, N, S) np.array

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    plt.imshow(np.log(conn_matrix[:,:,0]), cmap='RdYlBu')
    plt.show()

def plot_network(adj, coords):
    """
    Plots `data` as a cloud of points (nodes) with edges in 3D space based on specified `coords`

    Parameters
    ----------
    adj : (N,N) array_like
        Adjacency matrix for an `N` node parcellation; determines edges to plot
    coords : (N, 3) array_like
        x, y, z coordinates for `N` node parcellation

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
    """
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121, projection='3d')
    # remove nans from adj
    adj[np.isnan(adj)] = 0
    # Identify edges in the network
    edges = np.where(adj != 0)
    print(edges[0].shape)
    edge_cmap = plt.get_cmap('RdYlBu')
    #norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges].flatten()), vmax=np.max(adj[edges].flatten()))
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    edge_val = edge_cmap(norm(adj[edges].flatten()))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=c, alpha=0.5, zorder=0, linewidth=1)
    #scatter = ax.scatter(coords[:, 0],coords[:, 1],coords[:, 2], c=np.arange(1, 247), alpha=0.8)
    ax.view_init(elev=0, azim=180)
    ax.set_box_aspect((np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])))
    ax.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    # Identify edges in the network
    # edges = np.where(adj != 0)
    # edge_cmap = plt.get_cmap('RdYlBu')
    # norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges].flatten()), vmax=np.max(adj[edges].flatten()))
    # edge_val = edge_cmap(norm(adj[edges].flatten()))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax2.plot([x1, x2], [y1, y2], [z1, z2], color=c, alpha=0.5, zorder=0, linewidth=1)
    # Filter coords and node_colors based on the presence of edges
    filtered_coords = coords[edges[0]]

    ax2.scatter(filtered_coords[:, 0], filtered_coords[:, 1], filtered_coords[:, 2],
                            c='gray', alpha=0.8, s=5)
    ax2.view_init(elev=90, azim=-90)
    ax2.axis('off')
    ax2.set_box_aspect((np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=edge_cmap))
    fig.tight_layout()
    plt.show()


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

    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(
        list(df_con_v1.groupby(['subject']).apply(lambda x: x.set_index(['subject', 'roi']).to_numpy())))


    #connectivity_matrix_viewer(np_con_v1)
    plot_network(np_con_v1[:,:,0], load_brainnetome_centroids())
    #print(degrees_und(np_con_v1[:,:,:]))
    plot_point_brain(degrees_und(np_con_v1[:,:,0]), load_brainnetome_centroids(), views='ax', views_size=(8,4.8))
    plt.show()




if __name__ == "__main__":
    main()