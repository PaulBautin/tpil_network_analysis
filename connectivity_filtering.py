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
# Prerequis: environnement virtuel avec python, pandas, numpy et matplotlib (env_tpil)
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



def load_brainnetome_centroids():
    bn_centroids = get_centroids("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz")
    return bn_centroids

def distance_dependant_filter(np_con_v1):
    bn_centroids = load_brainnetome_centroids()
    eu_distance = squareform(pdist(bn_centroids, metric="euclidean"))
    hemiid = np.arange(1, 247) % 2
    consensus_conn = struct_consensus(np_con_v1, distance=eu_distance, hemiid=hemiid.reshape(-1, 1))
    print("\naverage_distance_without_filter: {}".format(np.mean(eu_distance, where=(eu_distance != 0))))
    print("average_distance_with_filter: {}".format(np.mean(eu_distance * consensus_conn, where=(consensus_conn != 0))))


def threshold_filter(np_con_v1):
    bn_centroids = load_brainnetome_centroids()
    eu_distance = squareform(pdist(bn_centroids, metric="euclidean"))
    threshold_conn = threshold_network(np.mean(np_con_v1,axis=2), retain=10)
    print("\naverage_distance_without_filter: {}".format(np.mean(eu_distance, where=(eu_distance != 0))))
    print("average_distance_with_filter: {}".format(np.mean(eu_distance * threshold_conn, where=(threshold_conn != 0))))

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

def scilpy_filter(df_connectivity_matrix):
    mask_con_sc = np.load('/home/mafor/dev_tpil/tpil_network_analysis/results/results_connectflow/con_mask_streamline.npy')[:-3,:-3]
    mask_clbp_sc = np.load('/home/mafor/dev_tpil/tpil_network_analysis/results/results_connectflow/clbp_mask_streamline.npy')[:-3,:-3]
    mask_con_len = np.load('/home/mafor/dev_tpil/tpil_network_analysis/results/results_connectflow/con_mask_len.npy')[:-3,:-3]
    mask_clbp_len = np.load('/home/mafor/dev_tpil/tpil_network_analysis/results/results_connectflow/clbp_mask_len.npy')[:-3,:-3]
    mask = mask_con_len * mask_clbp_len
    mask_data = df_connectivity_matrix * mask
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

    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(list(df_con_v1.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    #plt.imshow(np_con_v1[:,:,0])
    #plt.show()

    dist_filter = distance_dependant_filter(np_con_v1)

    thresh_filter = threshold_filter(np_con_v1)

    scil_filter = scilpy_filter(np_con_v1)

if __name__ == "__main__":
    main()