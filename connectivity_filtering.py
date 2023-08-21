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
import argparse
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
from connectivity_read_files import find_files_with_common_name
from netneurotools.utils import get_centroids
from netneurotools.networks import threshold_network, struct_consensus
from scipy.spatial.distance import squareform, pdist



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



def load_brainnetome_centroids(image="/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/labels/sub-pl007_ses-v1__nativepro_seg_all.nii.gz"):
    """
    Loads Euclidean coordinates of nodes
    Parameters
    ----------
    image : nifti label image

    Returns
    -------
    bn_centroids : (3, N) np.array node centroid coordinates
    """
    bn_centroids = get_centroids(image)
    return bn_centroids

def distance_dependant_filter(df_connectivity_matrix):
    """
    Calculates distance-dependent group consensus structural connectivity graph
    This filter is considered distance-dependent as it will divide data into bins (based on edge length percentile).
    It then takes the edge with the best consensus (edge represented with the most stability accross subjects)

    Betzel, R. F., Griffa, A., Hagmann, P., & Mišić, B. (2018). Distance- dependent consensus thresholds for generating 
    group-representative structural brain networks. Network Neuroscience, 1-22.

    Parameters
    ----------
    df_connectivity_matrix : (N, NxS) connectivity matrix pandas Dataframe, where N is the number of nodes and S the number of subjects
    Returns
    -------
    consensus_conn : (N, N) binary np.array matrix mask
    """
    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(list(df_connectivity_matrix.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    bn_centroids = load_brainnetome_centroids()
    eu_distance = squareform(pdist(bn_centroids, metric="euclidean"))
    # pairwise distance matrix, (N, 1) vector identifying right hemisphere (0)and left hemisphere (1) first has 8 on left and 7 on right
    hemiid = np.append(np.append(np.arange(0, 210) % 2, np.zeros(8)), np.ones(7))
    consensus_conn = struct_consensus(np_con_v1, distance=eu_distance, hemiid=hemiid.reshape(-1, 1))
    print("\naverage_distance_without_filter: {}".format(np.mean(eu_distance, where=(eu_distance != 0))))
    print("average_distance_with_filter: {}".format(np.mean(eu_distance * consensus_conn, where=(consensus_conn != 0))))
    return consensus_conn


def threshold_filter(df_connectivity_matrix):
    """
    Uses a minimum spanning tree to ensure that no nodes are disconnected from the resulting thresholded graph.
    Keeps top 10% of connections

    Parameters
    ----------
    df_connectivity_matrix : (N, N) connectivity matrix, pandas Dataframe, percent connection to retain

    Returns
    -------
    threshold_conn : (N, N) binary np.array matrix mask
    """
    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(list(df_connectivity_matrix.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    bn_centroids = load_brainnetome_centroids()
    eu_distance = squareform(pdist(bn_centroids, metric="euclidean"))
    threshold_conn = threshold_network(np.mean(np_con_v1,axis=2), retain=10)
    print("\naverage_distance_without_filter: {}".format(np.mean(eu_distance, where=(eu_distance != 0))))
    print("average_distance_with_filter: {}".format(np.mean(eu_distance * threshold_conn, where=(threshold_conn != 0))))
    return threshold_conn

#def filter_no_connections(df_connectivity_matrix):
    #df_con_sc = find_files_with_common_name(path_results_con, "sc.csv")
    #df_clbp_sc = find_files_with_common_name(path_results_con, "sc.csv")
    #df_con_sc_v1 = df_con[df_con_sc['session'] == "v1"].drop(["subject", "session"], axis=1)
    #df_clbp_sc_v1 = df_clbp[df_clbp_sc['session'] == "v1"].drop(["subject", "session"], axis=1)
    #df_connectivity_matrix.iloc[:, 1:] = df_connectivity_matrix.iloc[:, 1:].astype(float) # converts in format that np can use
    #df_zero_connections = []
    #df_zero_matrix = np.zeros_like(df_connectivity_matrix.iloc[:, 1:]) 
    #for row in range(df_zero_matrix.shape[0]):
    #    for col in range(df_zero_matrix.shape[1]):
    #        if df_connectivity_matrix.iloc[row, col+1] < 1: # all non-zero values (integers) convert to 1
    #            df_zero_matrix[row, col] = 0
    #            df_zero_connections.append((row, col))
    #        else:
    #            df_zero_matrix[row, col] = 1
    #If mean of ROI < 1, has at least one 0
    #df_mean_matrix = np.mean(df_zero_matrix.reshape(-1, 246, 246), axis=0)
    #df_mean_matrix[df_mean_matrix < 1] = 0
    #return df_mean_matrix

def scilpy_filter(df_connectivity_matrix, session):
    """
    Each node with a value of 1 represents a node with at least 90% of the population having at least 1 streamline
    and at least 90% of the population having at least 20mm of average streamlines length. 
    Population is all clbp and control subjects

    Parameters
    ----------
    df_connectivity_matrix :  (N, N) connectivity matrix pandas Dataframe, where N is the number of nodes 

    Returns
    -------
    mask_data : (N, N) filtered conenctivity matrix, pandas DataFrame
    """
    mask_v1_con_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v1_mask_sc.npy')
    mask_v1_clbp_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_sc.npy')
    mask_v1_con_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v1_mask_len.npy')
    mask_v1_clbp_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v1_mask_len.npy')
    mask_v2_con_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v2_mask_sc.npy')
    mask_v2_clbp_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_sc.npy')
    mask_v2_con_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v2_mask_len.npy')
    mask_v2_clbp_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v2_mask_len.npy')
    mask_v3_con_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v3_mask_sc.npy')
    mask_v3_clbp_sc = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_sc.npy')
    mask_v3_con_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/con_v3_mask_len.npy')
    mask_v3_clbp_len = np.load('/home/mafor/dev_tpil/tpil_networks/tpil_network_analysis/results/scilpy_filters/clbp_v3_mask_len.npy')
    df_mult = df_connectivity_matrix.set_index(["subject","roi"]) # step necessary to apply z-score equation on every subject
    valid_sessions = ['v1', 'v2', 'v3', 'all']
    if session not in valid_sessions:
        print("Invalid session. Valid sessions:", valid_sessions)
        return None
    elif session == 'v1':
        mask_v1 = mask_v1_con_len * mask_v1_clbp_len * mask_v1_con_sc * mask_v1_clbp_sc
        mask_data = df_mult * mask_v1
    elif session == 'v2':
        mask_v2 = mask_v2_con_len * mask_v2_clbp_len * mask_v2_con_sc * mask_v2_clbp_sc
        mask_data = df_mult * mask_v2
    elif session == 'v3':
        mask_v3 = mask_v3_con_len * mask_v3_clbp_len * mask_v3_con_sc * mask_v3_clbp_sc
        mask_data = df_mult * mask_v3
    elif session == 'all':
        mask_all = mask_v1_con_len * mask_v1_clbp_len * mask_v1_con_sc * mask_v1_clbp_sc * mask_v2_con_len * mask_v2_clbp_len * mask_v2_con_sc * mask_v2_clbp_sc * mask_v3_con_len * mask_v3_clbp_len * mask_v3_con_sc * mask_v3_clbp_sc
        mask_data = df_mult * mask_all
    return mask_data

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

    #plt.imshow(np_con_v1[:,:,0])
    #plt.show()

    dist_filter = distance_dependant_filter(df_con_v1)
    
    thresh_filter = threshold_filter(df_con_v1)

    scil_filter = scilpy_filter(df_con_v1)

if __name__ == "__main__":
    main()