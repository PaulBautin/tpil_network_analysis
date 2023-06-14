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

def z_score(df_con, df_clbp):
    df_con_mean = df_con.groupby("roi").mean()
    df_con_std = df_con.groupby("roi").std()
    df_clbp = df_clbp.set_index(["subject","roi"])
    df_clbp_mean = df_clbp.groupby("roi").mean()
    df_anor = (df_clbp - df_con_mean) / df_con_std
    df_anor_mean = df_anor.groupby("roi").mean()
    return df_anor_mean

def binary_mask(df_z_score_v1):
    #df_abs_matrix = df_z_score_v1.groupby("roi").abs()
    #threshold = 2
    #df_binary_matrix = df_abs_matrix.groupby("roi").where(df_abs_matrix >= threshold, 1, 0)
    df_abs_matrix = np.abs(df_z_score_v1)
    threshold = 2
    df_binary_matrix = np.where(df_abs_matrix >= threshold, 1, 0)
    print(df_binary_matrix)
    return df_binary_matrix


def find_files_with_common_name(directory, common_name):

    file_paths = glob.glob(directory + '/*/Compute_Connectivity/' + common_name)
    n = range(len(file_paths))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(file_paths[i]))) : pd.read_csv(file_paths[i], header=None) for i in n}
    df_paths = pd.concat(dict_paths)
    df_paths = df_paths.reset_index().rename(columns={'level_0': 'participant_id', 'level_1': 'roi'})
    # df_paths = df_paths[df_paths['participant_id'].str.contains('_ses-v1')]
    df_paths[['subject', 'session']] = df_paths['participant_id'].str.rsplit('_ses-', 1, expand=True)
    df_paths = df_paths.drop("participant_id", axis=1)
    return df_paths
    
    
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
    df_z_score_v1 = z_score(df_con_v1, df_clbp_v1) 
    df_binary_z_score_v1 = binary_mask(df_z_score_v1)
    
    #print(df_z_score_v1)
    #plt.imshow(df_z_score_v1, cmap='bwr', norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=20))
    #plt.colorbar()
    #plt.show()
    
    
if __name__ == "__main__":
    main()

