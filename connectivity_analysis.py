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
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
from connectivity_read_files import find_files_with_common_name
from connectivity_filtering import scilpy_filter
from connectivity_filtering import load_brainnetome_centroids
from connectivity_filtering import distance_dependant_filter
from connectivity_filtering import threshold_filter
from connectivity_graphing import circle_graph
from connectivity_graphing import histogram
from connectivity_figures import plot_network
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

def z_score(df_con_v1, df_clbp_v1):
    """
    Returns mean z-score between clbp and con data for a given edge

    Parameters
    ----------
    df_con_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects
    df_clbp_v1 : (N,SxN) pandas DataFrame where N is the number of nodes and S is the number of subjects

    Returns
    -------
    df_anor_mean : (N, N) pandas DataFrame
    """
    df_con_mean = mean_matrix(df_con_v1)
    df_con_std = df_con_v1.groupby("roi").std()
    df_clbp = df_clbp_v1.set_index(["subject","roi"]) # step necessary to apply z-score equation on every subject
    df_anor = (df_clbp - df_con_mean) / df_con_std
    df_anor_mean = mean_matrix(df_anor)
    return df_anor_mean

def prepare_data(df_connectivity_matrix, absolute=True):
    """
    Returns Data in the appropriate format for figure functions

    Parameters
    ----------
    df_connectivity_matrix : (N, N) pandas DataFrame

    Returns
    -------
    np_connectivity_matrix : (N, N) array_like
    """
    df_connectivity_matrix[np.isnan(df_connectivity_matrix)] = 0 
    np_connectivity_matrix = df_connectivity_matrix.values
    if absolute:
        np_connectivity_matrix = np.triu(abs(np_connectivity_matrix))
    else:
        np_connectivity_matrix = np.triu(np_connectivity_matrix)
    return np_connectivity_matrix

 
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

    # work only on one session at a time
    df_con_v1 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)

    ### Get similarity data
    df_con_sim = find_files_with_common_name(path_results_con, "sim.csv")
    df_clbp_sim = find_files_with_common_name(path_results_clbp, "sim.csv")

    # work only on one session at a time
    df_con_sim_v1 = df_con_sim[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_sim_v1 = df_clbp_sim[df_clbp['session'] == "v1"].drop("session", axis=1)
    
    
    df_con_mean = mean_matrix(df_con_v1)
    #df_con_hist = histogram(df_con_mean)
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/con_hist.txt', df_con_hist, fmt='%1.3f')
    df_clbp_mean = mean_matrix(df_clbp_v1)
    #df_clbp_hist = histogram(df_clbp_mean)
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/clbp_hist.txt', df_clbp_hist, fmt='%1.3f')
    
    df_z_score_v1 = z_score(df_con_v1, df_clbp_v1)
    np_z_score_v1 = prepare_data(df_z_score_v1, absolute=False)
    # transform to 3d numpy array (N, N, S) with N nodes and S subjects
    np_con_v1 = np.dstack(list(df_con_v1.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    np_con_dist = distance_dependant_filter(np_con_v1)
    #np_con_thresh = threshold_filter(np_con_v1)
    np_clbp_v1 = np.dstack(list(df_clbp_v1.groupby(['subject']).apply(lambda x: x.set_index(['subject','roi']).to_numpy())))
    np_clbp_dist = distance_dependant_filter(np_clbp_v1)
    #np_clbp_thresh = threshold_filter(np_clbp_v1)
    mask = np_con_dist * np_clbp_dist
    figure_data = mask * np_z_score_v1
    plot_network(figure_data, load_brainnetome_centroids())
    
    
    #df_z_score_mask_hist = df_z_score_mask.values.flatten()
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/sim_filtery.csv', df_z_score_mask_hist, fmt='%1.3f')
    #df_graph_z_score_mask = circle_graph(df_z_score_mask)
    #plot_network(df_z_score_mask, load_brainnetome_centroids())
    #plt.show()

    #plt.imshow(df_z_score_dist, cmap='bwr', norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=20))
    #plt.colorbar()
    #plt.show()



if __name__ == "__main__":
    main()