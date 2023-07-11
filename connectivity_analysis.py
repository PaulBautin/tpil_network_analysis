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
import argparse
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
from connectivity_graph import circle
from connectivity_graph import histogram

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
    mean_matrix = df_connectivity_matrix.groupby("roi").mean() # returns mean value of all subjects for an edge
    return mean_matrix

def z_score(df_con_v1, df_clbp_v1):
    df_con_mean = mean_matrix(df_con_v1)
    df_con_std = df_con_v1.groupby("roi").std()
    df_clbp = df_clbp_v1.set_index(["subject","roi"]) # step necessary to apply z-score equation on every subject
    df_anor = (df_clbp - df_con_mean) / df_con_std
    df_anor_mean = mean_matrix(df_anor)
    return df_anor_mean

def binary_mask(df_connectivity_matrix): # filters out for desired metrics
    df_abs_matrix = np.abs(df_connectivity_matrix)
    df_binary_matrix = (df_abs_matrix > 10.544).astype(np.int_) # threshold application (hard-coded)
    #np.where(df_abs_matrix > threshold, upper, lower)
    return df_binary_matrix

 
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
    

    
    df_con_mean = mean_matrix(df_con_v1)
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/con_mean_hist.txt', df_con_mean, fmt='%1.3f')
    df_con_binary = binary_mask(df_con_mean)

    df_clbp_mean = mean_matrix(df_clbp_v1)
    #df_clbp_hist = histogram(df_clbp_mean)
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/clbp_filter.txt', df_clbp_mean, fmt='%1.3f')
    #df_clbp_binary = binary_mask(df_clbp_mean)
    
    df_z_score_v1 = z_score(df_con_v1, df_clbp_v1)
    df_z_score_v1 = scilpy_filter(df_z_score_v1) 
    #df_z_score_v1[df_z_score_v1 < 1.96] = 0
    #df_z_score_hist = histogram(df_z_score_v1)
    #np.savetxt('/home/mafor/dev_tpil/tpil_network_analysis/data/z_filter.csv', df_z_score_hist, fmt='%1.3f')
    #df_graph_z_score_v1 = circle(df_z_score_v1)
    #plt.show()

    #plt.imshow(df_z_score_v1, cmap='bwr', norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=20))
    #plt.colorbar()
    #plt.show()
    
if __name__ == "__main__":
    main()