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
import bct
from connectivity_read_files import find_files_with_common_name

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

def compute_degree(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.values
    centrality = bct.degrees_und(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_betweenness(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.values
    centrality = bct.betweenness_wei(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_eigenvector(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.values
    centrality = bct.eigenvector_centrality_und(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

def compute_cluster(df_connectivity_matrix):
    np_connectivity_matrix = df_connectivity_matrix.values
    centrality = bct.clustering_coef_wu(np_connectivity_matrix)
    df_centrality = pd.DataFrame(centrality, columns=['centrality'])
    return df_centrality

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
    ### work only on one session at a time
    df_con_v1 = df_con[df_con['session'] == "v1"].drop("session", axis=1)
    df_clbp_v1 = df_clbp[df_clbp['session'] == "v1"].drop("session", axis=1)
    df_con_v2 = df_con[df_con['session'] == "v2"].drop("session", axis=1)
    df_clbp_v2 = df_clbp[df_clbp['session'] == "v2"].drop("session", axis=1)
    df_con_v3 = df_con[df_con['session'] == "v3"].drop("session", axis=1)
    df_clbp_v3 = df_clbp[df_clbp['session'] == "v3"].drop("session", axis=1)

    if __name__ == "__main__":
        main()